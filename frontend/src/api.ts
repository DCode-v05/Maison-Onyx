import type { InfoResponse, InspectResponse } from "./types";

const BASE = "/api";

export async function fetchInfo(): Promise<InfoResponse> {
  const res = await fetch(`${BASE}/info`);
  if (!res.ok) throw new Error(`info ${res.status}`);
  return res.json();
}

export async function inspect(
  master: File,
  live: File
): Promise<InspectResponse> {
  const fd = new FormData();
  fd.append("master", master);
  fd.append("live", live);
  const res = await fetch(`${BASE}/inspect`, {
    method: "POST",
    body: fd,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`inspect ${res.status}: ${text}`);
  }
  return res.json();
}
