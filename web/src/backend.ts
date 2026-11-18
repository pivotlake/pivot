// Talking to the web backend (`web/server`), which reads the pivot server
// for the page.

// What the backend answered, or an error saying why it could not.
async function readAnswer<Answer>(response: Response): Promise<Answer> {
  if (response.ok) {
    return response.json();
  }
  // The backend answers what it could not do with the reason, as JSON.
  // Anything else means the backend itself did not answer.
  if (response.headers.get("content-type")?.includes("application/json")) {
    const failure: { error: string } = await response.json();
    throw new Error(failure.error);
  }
  throw new Error(
    `the web backend is not answering (HTTP ${response.status}); start it with \`cargo run\` in web/server`,
  );
}

// Asks the backend for one of its readings of the pivot server.
export async function readBackend<Reading>(path: string): Promise<Reading> {
  return readAnswer(await fetch(path));
}

// Hands the backend `request` to act on, and returns what came of it.
export async function postBackend<Answer>(path: string, request: unknown): Promise<Answer> {
  return readAnswer(
    await fetch(path, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(request),
    }),
  );
}
