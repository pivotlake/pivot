import { useCallback, useEffect, useRef, useState } from "react";

/// Accumulating pager for "load more on scroll". Resets when `key` changes
/// (e.g. a different table is selected) and exposes `loadMore` to call when the
/// scroll container nears its bottom. `loader` is read through a ref so an
/// inline arrow doesn't churn the callback identity.
export function useInfiniteScroll<T>(
  key: string | null,
  pageSize: number,
  loader: (offset: number, limit: number) => Promise<{ items: T[]; hasMore: boolean }>,
) {
  const [items, setItems] = useState<T[]>([]);
  const [hasMore, setHasMore] = useState(false);
  const [loading, setLoading] = useState(false);
  const state = useRef({ offset: 0, loading: false, hasMore: false, key });
  const loaderRef = useRef(loader);
  loaderRef.current = loader;

  const loadMore = useCallback(async () => {
    const s = state.current;
    if (s.loading || !s.hasMore || s.key == null) return;
    s.loading = true;
    setLoading(true);
    const myKey = s.key;
    try {
      const page = await loaderRef.current(s.offset, pageSize);
      if (state.current.key !== myKey) return; // selection changed mid-flight
      state.current.offset += page.items.length;
      state.current.hasMore = page.hasMore && page.items.length > 0;
      setItems((prev) => [...prev, ...page.items]);
      setHasMore(state.current.hasMore);
    } catch {
      state.current.hasMore = false;
      setHasMore(false);
    } finally {
      state.current.loading = false;
      if (state.current.key === myKey) setLoading(false);
    }
  }, [pageSize]);

  // Reset and load the first page whenever the key changes.
  useEffect(() => {
    state.current = { offset: 0, loading: false, hasMore: key != null, key };
    setItems([]);
    setHasMore(key != null);
    setLoading(false);
    if (key != null) loadMore();
  }, [key, loadMore]);

  return { items, hasMore, loading, loadMore };
}
