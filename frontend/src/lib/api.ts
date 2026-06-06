import type { Flower, FlowerCreate } from "@/types/flower";

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text}`);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}

export const api = {
  flowers: {
    list: (status?: string): Promise<Flower[]> => {
      const qs = status ? `?status=${encodeURIComponent(status)}` : "";
      return request<Flower[]>(`/flowers${qs}`);
    },
    get: (id: number): Promise<Flower> => request<Flower>(`/flowers/${id}`),
    create: (body: FlowerCreate): Promise<Flower> =>
      request<Flower>("/flowers", { method: "POST", body: JSON.stringify(body) }),
    delete: (id: number): Promise<void> => request<void>(`/flowers/${id}`, { method: "DELETE" }),
    runData: (id: number): Promise<Flower> =>
      request<Flower>(`/flowers/${id}/data`, { method: "POST" }),
    runImages: (id: number): Promise<Flower> =>
      request<Flower>(`/flowers/${id}/images`, { method: "POST" }),
  },
  export: {
    xcassets: (): Promise<{ exported: number; output_path: string }> =>
      request<{ exported: number; output_path: string }>("/export", { method: "POST" }),
  },
};
