export type ExperimentResult = {
  heterophily: number;
  cutoffs: number[];
  low_pass_accuracy: number[];
  high_pass_accuracy: number[];
  baseline_accuracy: number;
};

export async function runExperiment(): Promise<ExperimentResult> {
  const res = await fetch("http://127.0.0.1:8000/run");
  if (!res.ok) {
    throw new Error("Failed to fetch experiment results");
  }
  return res.json();
}
