/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * The OpenSearch Contributors require contributions made to
 * this file be licensed under the Apache-2.0 license or a
 * compatible open source license.
 */

package org.opensearch.telemetry.tracing.sampler;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import org.opensearch.SpecialPermission;
import org.opensearch.common.settings.Settings;
import org.opensearch.telemetry.OTelTelemetrySettings;
import org.opensearch.telemetry.TelemetrySettings;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;
import java.util.List;
import java.util.ListIterator;

import io.opentelemetry.sdk.trace.samplers.Sampler;

/**
 * Factory class to create the instance of OTelSampler
 */
public class OTelSamplerFactory {

    /**
     * Logger instance for logging messages related to the OTelSamplerFactory.
     */
    private static final Logger logger = LogManager.getLogger(OTelSamplerFactory.class);

    /**
     * Base constructor.
     */
    private OTelSamplerFactory() {

    }

    /**
     * Creates the {@link Sampler} instances based on the TRACER_SPAN_SAMPLER_CLASSES value.
     *
     * @param telemetrySettings TelemetrySettings.
     * @param setting           Settings
     * @return list of samplers.
     */
    public static Sampler create(TelemetrySettings telemetrySettings, Settings setting) {
        List<String> samplersNameList = OTelTelemetrySettings.OTEL_TRACER_SPAN_SAMPLER_CLASS_SETTINGS.get(setting);
        ListIterator<String> li = samplersNameList.listIterator(samplersNameList.size());

        Sampler fallbackSampler = null;

        // Iterating samplers list in reverse order to create chain of sampler
        while (li.hasPrevious()) {
            String samplerName = li.previous();
            fallbackSampler = instantiateSampler(samplerName, telemetrySettings, fallbackSampler);
        }

        return fallbackSampler;
    }

    private static Sampler instantiateSampler(String samplerClassName, TelemetrySettings telemetrySettings, Sampler fallbackSampler) {
        try {
            // Check we ourselves are not being called by unprivileged code.
            SpecialPermission.check();

            return AccessController.doPrivileged((PrivilegedExceptionAction<Sampler>) () -> {
                try {
                    Class<?> samplerClass = Class.forName(samplerClassName);

                    // Define the method type which receives TelemetrySettings & Sampler as arguments
                    MethodType methodType = MethodType.methodType(Sampler.class, TelemetrySettings.class, Sampler.class);

                    return (Sampler) MethodHandles.publicLookup()
                        .findStatic(samplerClass, "create", methodType)
                        .invokeExact(telemetrySettings, fallbackSampler);
                } catch (Throwable e) {
                    if (e.getCause() instanceof NoSuchMethodException) {
                        throw new IllegalStateException("No create method exist in [" + samplerClassName + "]");
                    } else {
                        throw new IllegalStateException("Sampler instantiation failed for class [" + samplerClassName + "]", e.getCause());
                    }
                }
            });
        } catch (Exception e) {
            throw new IllegalStateException("Sampler instantiation failed for class [" + samplerClassName + "]", e.getCause());
        }
    }
}
