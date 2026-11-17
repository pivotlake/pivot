// Asks the web backend (`web/server`) for one of its readings of the pivot
// server.
export async function readBackend<Reading>(path: string): Promise<Reading> {
  const response = await fetch(path);
  if (response.ok) {
    return response.json();
  }
  // The backend answers a read it could not make with the reason, as JSON.
  // Anything else means the backend itself did not answer.
  if (response.headers.get("content-type")?.includes("application/json")) {
    const failure: { error: string } = await response.json();
    throw new Error(failure.error);
  }
  throw new Error(
    `the web backend is not answering (HTTP ${response.status}); start it with \`cargo run\` in web/server`,
  );
}
