import { useEffect, useState } from "react";
import { QueryPage } from "./QueryPage";
import { ServerPage } from "./ServerPage";
import { TablesPage } from "./TablesPage";

// The page a location's hash names: `#/tables` and `#/tables/<table id>` are
// the tables page, `#/query` the query page, and anything else the server
// page.
function readRoute(): { page: "server" } | { page: "tables"; tableId?: string } | { page: "query" } {
  const [, page, tableId] = window.location.hash.split("/");
  if (page === "tables") return { page: "tables", tableId };
  if (page === "query") return { page: "query" };
  return { page: "server" };
}

export function App() {
  const [route, setRoute] = useState(readRoute);
  useEffect(() => {
    const followHash = () => setRoute(readRoute());
    window.addEventListener("hashchange", followHash);
    return () => window.removeEventListener("hashchange", followHash);
  }, []);

  return (
    <>
      <header className="app-header">
        <div className="wordmark">
          pivot
          <span className="wordmark-cursor" aria-hidden="true" />
        </div>
        <nav className="app-nav">
          <a href="#/" aria-current={route.page === "server" ? "page" : undefined}>
            Server
          </a>
          <a href="#/tables" aria-current={route.page === "tables" ? "page" : undefined}>
            Tables
          </a>
          <a href="#/query" aria-current={route.page === "query" ? "page" : undefined}>
            Query
          </a>
        </nav>
      </header>
      <main className="app-page">
        {route.page === "server" && <ServerPage />}
        {route.page === "tables" && <TablesPage selectedTableId={route.tableId} />}
        {route.page === "query" && <QueryPage />}
      </main>
    </>
  );
}
