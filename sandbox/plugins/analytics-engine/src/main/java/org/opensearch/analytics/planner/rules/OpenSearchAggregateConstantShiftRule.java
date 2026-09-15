/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.analytics.planner.rules;

import org.apache.calcite.plan.RelOptRule;
import org.apache.calcite.plan.RelOptRuleCall;
import org.apache.calcite.rel.RelCollations;
import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.core.Aggregate;
import org.apache.calcite.rel.core.AggregateCall;
import org.apache.calcite.rel.logical.LogicalAggregate;
import org.apache.calcite.rel.logical.LogicalProject;
import org.apache.calcite.rel.type.RelDataType;
import org.apache.calcite.rex.RexBuilder;
import org.apache.calcite.rex.RexCall;
import org.apache.calcite.rex.RexLiteral;
import org.apache.calcite.rex.RexNode;
import org.apache.calcite.rex.RexUtil;
import org.apache.calcite.sql.SqlAggFunction;
import org.apache.calcite.sql.SqlKind;
import org.apache.calcite.sql.fun.SqlStdOperatorTable;
import org.apache.calcite.sql.type.SqlTypeName;
import org.apache.calcite.tools.RelBuilder;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Pre-marking rule: hoists a constant out of an aggregate's argument so the arithmetic runs once on the
 * aggregated value instead of once per row, and every term over the same column shares one set of
 * accumulators. For an integer column {@code x} and an integer literal {@code k}:
 *
 * <pre>
 *   SUM(x ± k)   → SUM(x) ± k * COUNT(x)
 *   COUNT(x ± k) → COUNT(x)
 *   MIN(x ± k)   → MIN(x) ± k          MAX(x ± k) → MAX(x) ± k
 *
 *   stats sum(x), sum(x + 1), sum(x + 2), min(x + 3), count(x + 4)
 *
 *   Aggregate(SUM($0), SUM($1), SUM($2), MIN($3), COUNT($4))     ← 5 accumulators
 *     Project(x, x + 1, x + 2, x + 3, x + 4)                       ← 4 additions per row
 *
 *   Project($0, $0 + $1, $0 + 2 * $1, $2 + 3, $1)                  ← arithmetic once, on aggregated values
 *     Aggregate(SUM($0), COUNT($0), MIN($0))                       ← 3 accumulators
 *       Project(x)                                                 ← 1 column read
 * </pre>
 *
 * <p>Runs in {@code aggregate-decompose}, before marking and before the PARTIAL / FINAL split, so both halves
 * of a distributed aggregate see the reduced call list. It cannot be left to a backend: by the time a
 * fragment is handed over, the argument is an opaque projected column, not an expression. Shares a rule
 * collection with the AVG / STDDEV reduce rule so {@code AVG(x ± k)}, reduced to {@code SUM(x ± k) /
 * COUNT(x ± k)}, collapses in the same fixpoint loop.
 *
 * <p>{@code COUNT(x)}, not {@code COUNT(*)}, keeps the identities exact with NULLs and empty groups.
 * Integer-only: float addition is not associative. {@code x} and {@code CAST(x):BIGINT} count as the same
 * column (the widest variant is aggregated, results are cast back), which is what lets {@code sum(x)} and
 * {@code sum(x + 1)} share an accumulator. Output names and types are preserved. Fires only when at least one
 * call is a supported function over {@code x ± k} with a non-zero integer {@code k}; DISTINCT, FILTER,
 * approximate and multi-argument calls never take part.
 *
 * @opensearch.internal
 */
public class OpenSearchAggregateConstantShiftRule extends RelOptRule {

    // Aggregates for which fn(x ± k) can be computed from fn(x) (plus COUNT(x) for SUM).
    private static final Set<SqlAggFunction> SUPPORTED_FUNCTIONS = Set.of(
        SqlStdOperatorTable.SUM,
        SqlStdOperatorTable.COUNT,
        SqlStdOperatorTable.MIN,
        SqlStdOperatorTable.MAX
    );

    public OpenSearchAggregateConstantShiftRule() {
        super(operand(LogicalAggregate.class, operand(LogicalProject.class, any())), "OpenSearchAggregateConstantShiftRule");
    }

    /**
     * Cheap structural precheck. Kept separate from {@link #onMatch} on purpose: Calcite records a rule attempt
     * (and {@code RuleProfilingListener} reports it) only when this returns true, so aggregates with nothing
     * to hoist must be declined here, not inside {@code onMatch}.
     */
    @Override
    public boolean matches(RelOptRuleCall call) {
        LogicalAggregate aggregate = call.rel(0);
        LogicalProject project = call.rel(1);

        // Grouping sets would need one groupKey per set; pre-operand (rexList) calls are the one thing
        // RelBuilder's input pruning does not remap. Neither shape reaches us from PPL today.
        if (aggregate.getGroupType() != Aggregate.Group.SIMPLE) {
            return false;
        }
        for (AggregateCall ac : aggregate.getAggCallList()) {
            if (!ac.rexList.isEmpty()) {
                return false;
            }
        }
        // At least one supported call over x ± k with k != 0. Also what stops the rule from re-firing on its
        // own output, whose calls are all shift 0.
        for (AggregateCall ac : aggregate.getAggCallList()) {
            ShiftedCall sc = ShiftedCall.of(ac, project);
            if (sc != null && sc.shift().signum() != 0) {
                return true;
            }
        }
        return false;
    }

    /** {@code Project(recomposed outputs) / Aggregate(shared primitives) / Project(original columns + bases)}. */
    @Override
    public void onMatch(RelOptRuleCall call) {
        LogicalAggregate aggregate = call.rel(0);
        LogicalProject project = call.rel(1);
        RelBuilder relBuilder = call.builder();

        ShiftedCall[] shifted = classify(aggregate, project);
        InputProject input = projectWithBases(relBuilder, project, unifyBases(shifted));
        CallRegistry registry = new CallRegistry(aggregate.getCluster().getRexBuilder(), input.node(), aggregate.getGroupCount());
        List<RexNode> outputs = recomposeOutputs(aggregate, shifted, input.baseSlot(), registry);

        call.transformTo(
            relBuilder.aggregate(relBuilder.groupKey(aggregate.getGroupSet()), registry.calls())
                .project(outputs, aggregate.getRowType().getFieldNames(), true)
                .build()
        );
    }

    // ---- Rewrite steps ----

    /** Classifies every call once; {@code null} where a call does not take part (kept as-is, remapped by RelBuilder). */
    private static ShiftedCall[] classify(LogicalAggregate aggregate, LogicalProject project) {
        List<AggregateCall> calls = aggregate.getAggCallList();
        ShiftedCall[] shifted = new ShiftedCall[calls.size()];
        for (int i = 0; i < calls.size(); i++) {
            shifted[i] = ShiftedCall.of(calls.get(i), project);
        }
        return shifted;
    }

    /** {@code x}, {@code CAST(x):BIGINT} and {@code CAST(x):INTEGER} are one base; keep the variant the others widen into. */
    private static Map<String, RexNode> unifyBases(ShiftedCall[] shifted) {
        Map<String, RexNode> baseByKey = new HashMap<>();
        for (ShiftedCall sc : shifted) {
            if (sc != null) {
                baseByKey.merge(baseKey(sc.base()), sc.base(), OpenSearchAggregateConstantShiftRule::widerOf);
            }
        }
        return baseByKey;
    }

    /** The rewritten input Project (left on the builder's stack) and the slot each base column landed in. */
    private record InputProject(RelNode node, Map<String, Integer> baseSlot) {
    }

    /**
     * The original columns plus one per base. {@link RelBuilder#aggregate} later prunes the {@code x ± k} columns
     * nobody reads any more, so this does not have to.
     */
    private static InputProject projectWithBases(RelBuilder relBuilder, LogicalProject project, Map<String, RexNode> baseByKey) {
        List<RexNode> exprs = new ArrayList<>(project.getProjects());
        List<String> names = new ArrayList<>(project.getRowType().getFieldNames());
        Map<String, Integer> baseSlot = new HashMap<>();
        for (Map.Entry<String, RexNode> base : baseByKey.entrySet()) {
            int slot = exprs.indexOf(base.getValue());
            if (slot < 0) {
                slot = exprs.size();
                exprs.add(base.getValue());
                names.add(null);
            }
            baseSlot.put(base.getKey(), slot);
        }
        return new InputProject(relBuilder.push(project.getInput()).project(exprs, names, true).peek(), baseSlot);
    }

    /**
     * One expression per original output column: group keys pass through; aggregates are recomposed from the
     * shared primitives or, when they do not take part, re-registered unchanged. Output types are restored either way.
     */
    private static List<RexNode> recomposeOutputs(
        LogicalAggregate aggregate,
        ShiftedCall[] shifted,
        Map<String, Integer> baseSlot,
        CallRegistry registry
    ) {
        int groupCount = aggregate.getGroupCount();
        List<RexNode> outputs = new ArrayList<>();
        for (int key = 0; key < groupCount; key++) {
            outputs.add(registry.rexBuilder().makeInputRef(outputType(aggregate, key), key));
        }
        List<AggregateCall> originals = aggregate.getAggCallList();
        for (int i = 0; i < originals.size(); i++) {
            ShiftedCall sc = shifted[i];
            RexNode value = sc == null ? registry.register(originals.get(i)) : recompose(sc, baseSlot.get(baseKey(sc.base())), registry);
            outputs.add(castTo(outputType(aggregate, groupCount + i), value, registry.rexBuilder()));
        }
        return outputs;
    }

    /**
     * {@code fn(base ± k)} from the shared {@code fn(base)}: SUM shifts by {@code k} per counted row, MIN / MAX by
     * {@code k} once, COUNT is shift-invariant. A shift-0 call is the shared primitive itself.
     */
    private static RexNode recompose(ShiftedCall sc, int slot, CallRegistry registry) {
        RexNode aggregated = registry.primitive(sc.function(), slot);
        if (sc.shift().signum() == 0 || sc.function() == SqlStdOperatorTable.COUNT) {
            return aggregated;
        }
        BigDecimal k = sc.shift().abs();
        RexNode delta = sc.function() == SqlStdOperatorTable.SUM
            ? scaledCount(k, slot, registry)
            : registry.rexBuilder().makeExactLiteral(k);
        return registry.rexBuilder()
            .makeCall(sc.shift().signum() > 0 ? SqlStdOperatorTable.PLUS : SqlStdOperatorTable.MINUS, aggregated, delta);
    }

    /** {@code k * COUNT(base)}, or just {@code COUNT(base)} when {@code k == 1}. */
    private static RexNode scaledCount(BigDecimal k, int slot, CallRegistry registry) {
        RexNode count = registry.primitive(SqlStdOperatorTable.COUNT, slot);
        return k.compareTo(BigDecimal.ONE) == 0
            ? count
            : registry.rexBuilder().makeCall(SqlStdOperatorTable.MULTIPLY, registry.rexBuilder().makeExactLiteral(k), count);
    }

    /** The recomposition may widen (BIGINT sum over a SMALLINT base); restore the declared output type. */
    private static RexNode castTo(RelDataType type, RexNode value, RexBuilder rexBuilder) {
        return value.getType().equals(type) ? value : rexBuilder.makeCast(type, value, true);
    }

    private static RelDataType outputType(LogicalAggregate aggregate, int ordinal) {
        return aggregate.getRowType().getFieldList().get(ordinal).getType();
    }

    /**
     * The rewritten aggregate's call list. {@link RexBuilder#addAggCall} adds a call once and hands back a reference
     * to its output column, so N terms over one base cost one accumulator per primitive.
     */
    private static final class CallRegistry {
        private final RexBuilder rexBuilder;
        private final RelNode input;
        private final int groupCount;
        private final List<AggregateCall> calls = new ArrayList<>();
        private final Map<AggregateCall, RexNode> refs = new HashMap<>();

        CallRegistry(RexBuilder rexBuilder, RelNode input, int groupCount) {
            this.rexBuilder = rexBuilder;
            this.input = input;
            this.groupCount = groupCount;
        }

        /** Adds {@code call} unless an identical one is already registered; returns the reference to its output. */
        RexNode register(AggregateCall call) {
            return rexBuilder.addAggCall(call, groupCount, calls, refs, input::fieldIsNullable);
        }

        /** Registers plain {@code fn(input column slot)}, typed from {@code input}. */
        RexNode primitive(SqlAggFunction fn, int slot) {
            return register(newAggCall(fn, slot, groupCount, input));
        }

        List<AggregateCall> calls() {
            return calls;
        }

        RexBuilder rexBuilder() {
            return rexBuilder;
        }
    }

    // ---- Call classification ----

    /**
     * {@code function(base ± shift)}: an integer expression moved by an integer literal. A plain
     * {@code function(base)} is a {@code ShiftedCall} with {@code shift == 0} so it can share the base's
     * accumulators with its shifted siblings.
     */
    private record ShiftedCall(SqlAggFunction function, RexNode base, BigDecimal shift) {

        /** Recognizes {@code fn(x + k)}, {@code fn(k + x)}, {@code fn(x - k)} and plain integer {@code fn(x)}; null otherwise. */
        static ShiftedCall of(AggregateCall ac, LogicalProject project) {
            if (!SUPPORTED_FUNCTIONS.contains(ac.getAggregation())
                || ac.isDistinct()
                || ac.distinctKeys != null
                || ac.isApproximate()
                || ac.filterArg >= 0
                || ac.getArgList().size() != 1) {
                return null;
            }
            SqlAggFunction fn = ac.getAggregation();
            RexNode arg = project.getProjects().get(ac.getArgList().get(0));
            // Bases are unified by expression text, which is only sound for deterministic expressions: two
            // independent RAND() calls must stay two draws, not one shared column.
            if (!RexUtil.isDeterministic(arg)) {
                return null;
            }
            if (!(arg instanceof RexCall arithmetic)
                || arithmetic.getOperands().size() != 2
                || (arithmetic.getKind() != SqlKind.PLUS && arithmetic.getKind() != SqlKind.MINUS)) {
                // Not an arithmetic argument: a plain integer column takes part with shift 0.
                return isIntegerType(arg.getType()) ? new ShiftedCall(fn, arg, BigDecimal.ZERO) : null;
            }
            RexNode left = arithmetic.getOperands().get(0);
            RexNode right = arithmetic.getOperands().get(1);
            boolean minus = arithmetic.getKind() == SqlKind.MINUS;
            ShiftedCall shifted = null;
            if (isIntegerLiteral(right) && isIntegerType(left.getType())) {
                BigDecimal k = literalValue(right);
                shifted = new ShiftedCall(fn, left, minus ? k.negate() : k);
            } else if (!minus && isIntegerLiteral(left) && isIntegerType(right.getType())) {
                shifted = new ShiftedCall(fn, right, literalValue(left));
            }
            // k - x (needs -x, a scale), x + y (no literal), x + 1.5 (not integer): leave as written.
            if (shifted == null) {
                return null;
            }
            // Per-row x ± k wraps silently on overflow. SUM and COUNT survive that (wrapping addition commutes with
            // the identity), MIN / MAX do not: the order of wrapped values is not the order of the originals. Hoist
            // MIN / MAX only when no row can overflow, which the base type's range guarantees for small enough k.
            if ((fn == SqlStdOperatorTable.MIN || fn == SqlStdOperatorTable.MAX) && !fitsWithoutOverflow(shifted.base(), shifted.shift())) {
                return null;
            }
            return shifted;
        }
    }

    // ---- Base unification ----

    /** Identity of a base with lossless casts stripped, so {@code x} and {@code CAST(x):BIGINT} map to one key. */
    private static String baseKey(RexNode base) {
        RexNode node = base;
        while (node.getKind() == SqlKind.CAST && RexUtil.isLosslessCast(node)) {
            node = ((RexCall) node).getOperands().get(0);
        }
        return node.toString();
    }

    /** Of two variants of one base, the one the other converts into without loss. */
    private static RexNode widerOf(RexNode current, RexNode candidate) {
        return RexUtil.isLosslessCast(current.getType(), candidate.getType()) ? candidate : current;
    }

    // ---- Small helpers ----

    /** Plain {@code fn(arg)}: no DISTINCT / FILTER / collation, type inferred from {@code input}. */
    private static AggregateCall newAggCall(SqlAggFunction fn, int arg, int groupCount, RelNode input) {
        return AggregateCall.create(
            fn,
            false,
            false,
            false,
            List.of(),
            List.of(arg),
            -1,
            null,
            RelCollations.EMPTY,
            groupCount,
            input,
            null,
            null
        );
    }

    /** A non-null integer literal, possibly wrapped in casts that keep it integer. */
    private static boolean isIntegerLiteral(RexNode node) {
        if (!RexUtil.isLiteral(node, true) || !isIntegerType(node.getType())) {
            return false;
        }
        RexLiteral literal = (RexLiteral) RexUtil.removeCast(node);
        return !literal.isNull() && isIntegerType(literal.getType());
    }

    private static BigDecimal literalValue(RexNode node) {
        return ((RexLiteral) RexUtil.removeCast(node)).getValueAs(BigDecimal.class);
    }

    private static boolean isIntegerType(RelDataType type) {
        return SqlTypeName.INT_TYPES.contains(type.getSqlTypeName());
    }

    /**
     * Whether {@code base ± shift} stays inside the 64-bit range for every possible value of {@code base}, judged
     * from the base's own type (lossless casts stripped): a SMALLINT or INTEGER column has room for any practical
     * {@code k}, a BIGINT column only for {@code k == 0}.
     */
    private static boolean fitsWithoutOverflow(RexNode base, BigDecimal shift) {
        RexNode node = base;
        while (node.getKind() == SqlKind.CAST && RexUtil.isLosslessCast(node)) {
            node = ((RexCall) node).getOperands().get(0);
        }
        BigDecimal bound = switch (node.getType().getSqlTypeName()) {
            case TINYINT -> BigDecimal.valueOf(Byte.MAX_VALUE + 1L);
            case SMALLINT -> BigDecimal.valueOf(Short.MAX_VALUE + 1L);
            case INTEGER -> BigDecimal.valueOf(Integer.MAX_VALUE + 1L);
            default -> BigDecimal.valueOf(Long.MAX_VALUE);
        };
        // |base| <= bound, so |base ± k| <= bound + |k| must not exceed Long.MAX_VALUE.
        return bound.add(shift.abs()).compareTo(BigDecimal.valueOf(Long.MAX_VALUE)) <= 0;
    }
}
