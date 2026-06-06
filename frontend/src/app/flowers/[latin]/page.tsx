"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { api } from "@/lib/api";
import { ConfidenceScoresView } from "@/components/ConfidenceScores";
import { DataFieldsView } from "@/components/DataFieldsView";
import type { Flower } from "@/types/flower";

export default function FlowerDetailPage() {
  const { latin } = useParams<{ latin: string }>();
  const latinName = decodeURIComponent(latin);

  const [flower, setFlower] = useState<Flower | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runningData, setRunningData] = useState(false);
  const [runningImages, setRunningImages] = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const all = await api.flowers.list();
      const found = all.find((f) => f.latin_name === latinName) ?? null;
      setFlower(found);
      setError(null);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleRunData = async () => {
    if (!flower) return;
    setRunningData(true);
    try {
      await api.flowers.runData(flower.id);
      await load();
    } catch (e) {
      setError(String(e));
    } finally {
      setRunningData(false);
    }
  };

  const handleRunImages = async () => {
    if (!flower) return;
    setRunningImages(true);
    try {
      await api.flowers.runImages(flower.id);
      await load();
    } catch (e) {
      setError(String(e));
    } finally {
      setRunningImages(false);
    }
  };

  if (loading) return <div className="text-sm text-gray-400 py-12 text-center">Loading...</div>;
  if (error) return <div className="text-sm text-red-600 py-12 text-center">{error}</div>;
  if (!flower) return <div className="text-sm text-gray-400 py-12 text-center">Flower not found.</div>;

  return (
    <div>
      <Link href="/" className="text-sm text-blue-600 hover:underline mb-6 block">← Back to library</Link>

      <div className="flex items-start justify-between gap-4 mb-8">
        <div>
          <h2 className="text-2xl font-semibold italic">{flower.latin_name}</h2>
          {flower.common_name && <p className="text-gray-500 mt-1">{flower.common_name}</p>}
          {flower.wikipedia_url && (
            <a href={flower.wikipedia_url} target="_blank" rel="noopener noreferrer"
               className="text-xs text-blue-600 hover:underline mt-1 block">
              Wikipedia
            </a>
          )}
        </div>

        <div className="flex gap-2 flex-shrink-0">
          <button
            onClick={handleRunData}
            disabled={runningData}
            className="text-sm px-4 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 transition-colors"
          >
            {runningData ? "Running data..." : "Run Data"}
          </button>
          <button
            onClick={handleRunImages}
            disabled={runningImages || !["enriched", "images_done", "complete"].includes(flower.status)}
            className="text-sm px-4 py-2 rounded-lg border border-gray-300 bg-white hover:bg-gray-50 disabled:opacity-50 transition-colors"
          >
            {runningImages ? "Running images..." : "Run Images"}
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          <DataFieldsView flower={flower} />
        </div>
        <div className="space-y-4">
          {/* Plant images */}
          {(flower.info_image_path || flower.main_image_path) && (
            <div className="rounded-xl border border-gray-200 bg-white p-5">
              <h3 className="text-sm font-semibold mb-3">Images</h3>
              <div className="space-y-3">
                {flower.info_image_path && (
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Info</p>
                    <img
                      src={`${process.env.NEXT_PUBLIC_API_URL ?? ""}/flowers/${flower.id}/images/info`}
                      alt={`${flower.latin_name} info`}
                      className="w-full rounded-lg border border-gray-100 object-cover"
                      style={{ maxHeight: "180px" }}
                    />
                    {flower.info_image_author && (
                      <p className="text-xs text-gray-400 mt-1 truncate">© {flower.info_image_author}</p>
                    )}
                  </div>
                )}
                {flower.main_image_path && (
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Transparent blossom</p>
                    <img
                      src={`${process.env.NEXT_PUBLIC_API_URL ?? ""}/flowers/${flower.id}/images/main`}
                      alt={`${flower.latin_name} blossom`}
                      className="w-full rounded-lg object-contain bg-gray-50"
                      style={{ maxHeight: "180px" }}
                    />
                  </div>
                )}
                {flower.lock_image_path && (
                  <div>
                    <p className="text-xs text-gray-400 mb-1">Lock screen</p>
                    <img
                      src={`${process.env.NEXT_PUBLIC_API_URL ?? ""}/flowers/${flower.id}/images/lock`}
                      alt={`${flower.latin_name} lock`}
                      className="w-full rounded-lg object-cover"
                      style={{ maxHeight: "100px" }}
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          {flower.petal_color_hex && (
            <div className="rounded-xl border border-gray-200 bg-white p-5">
              <h3 className="text-sm font-semibold mb-3">Petal Color</h3>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full border border-gray-200"
                     style={{ background: flower.petal_color_hex }} />
                <span className="font-mono text-sm text-gray-600">{flower.petal_color_hex}</span>
              </div>
            </div>
          )}

          {flower.confidence_scores && (
            <ConfidenceScoresView scores={flower.confidence_scores} />
          )}

          <div className="rounded-xl border border-gray-200 bg-white p-5">
            <h3 className="text-sm font-semibold mb-3">Status</h3>
            <p className="text-sm text-gray-600 capitalize">{flower.status}</p>
            {flower.feature_month && (
              <p className="text-xs text-gray-400 mt-2">
                Featured: {flower.feature_year}-{String(flower.feature_month).padStart(2, "0")}-{String(flower.feature_day).padStart(2, "0")}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
