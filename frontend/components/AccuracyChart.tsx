"use client";

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid,
  ReferenceLine,
} from "recharts";

type Props = {
  cutoffs: number[];
  low: number[];
  high: number[];
  baseline: number;
};

export default function AccuracyChart({
  cutoffs,
  low,
  high,
  baseline,
}: Props) {
  const data = cutoffs.map((c, i) => ({
    cutoff: c,
    low: low[i],
    high: high[i],
  }));

  return (
    <LineChart
        width={700}
        height={380}
        data={data}
        margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
        >

      <CartesianGrid strokeDasharray="3 3" />
      <XAxis
        dataKey="cutoff"
        label={{ value: "Spectral Cutoff (k)", position: "insideBottom", offset: -5 }}
        />

        <YAxis
        label={{ value: "Accuracy", angle: -90, position: "insideLeft" }}
        domain={[0.75, 1.0]}
        />

      <Tooltip />
      <Legend verticalAlign="top" height={36} />

      <Line
        type="monotone"
        dataKey="low"
        stroke="#2563eb"
        strokeWidth={3}
        dot={{ r: 4 }}
        name="Low-pass filtering"
        />

        <Line
        type="monotone"
        dataKey="high"
        stroke="#f97316"
        strokeWidth={3}
        dot={{ r: 4 }}
        name="High-pass filtering"
        />


      <ReferenceLine
        y={baseline}
        stroke="gray"
        strokeDasharray="4 4"
        />
    </LineChart>
  );
}
