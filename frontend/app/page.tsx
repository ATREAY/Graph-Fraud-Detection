"use client";

import { useState } from "react";
import AccuracyChart from "@/components/AccuracyChart";
import { runExperiment, ExperimentResult } from "@/lib/api";

export default function Home() {
  const [data, setData] = useState<ExperimentResult | null>(null);
  const [loading, setLoading] = useState(false);

  const handleRun = async () => {
    setLoading(true);
    try {
      const res = await runExperiment();
      setData(res);
    } catch (e) {
      console.error(e);
      alert("Failed to run experiment");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="min-h-screen bg-linear-to-br from-black via-gray-900 to-black flex items-center justify-center px-4">
      <div className="max-w-5xl w-full bg-white rounded-2xl shadow-2xl p-10">
        {/* Header */}
        <h1 className="text-3xl font-extrabold text-gray-900">
          Graph Fraud Detection under Heterophily
        </h1>
        <p className="text-gray-500 mt-2 mb-8">
          Spectral analysis of low-pass vs high-pass filtering (AAAI-style)
        </p>

        {/* Button */}
        <button
          onClick={handleRun}
          disabled={loading}
          className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 mb-8"
        >
          {loading ? "Running..." : "Run Experiment"}
        </button>

        {/* Metrics */}
        {data && (
          <>
            <div className="grid grid-cols-2 gap-6 mb-10">
              <div className="bg-gray-100 rounded-xl p-5">
                <p className="text-sm text-gray-500">Heterophily</p>
                <p className="text-3xl font-bold text-gray-900">
                  {data.heterophily.toFixed(3)}
                </p>
              </div>

              <div className="bg-gray-100 rounded-xl p-5">
                <p className="text-sm text-gray-500">Baseline Accuracy</p>
                <p className="text-3xl font-bold text-gray-900">
                  {data.baseline_accuracy.toFixed(3)}
                </p>
              </div>
            </div>

            {/* ✅ CHART (THIS WAS THE MISSING PART) */}
            <div className="flex justify-center">
              <AccuracyChart
                cutoffs={data.cutoffs}
                low={data.low_pass_accuracy}
                high={data.high_pass_accuracy}
                baseline={data.baseline_accuracy}
              />
            </div>
          </>
        )}
      </div>
    </main>
  );
}
