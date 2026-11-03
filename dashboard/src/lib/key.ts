import { useState } from "react";

const STORE = "itara-admin-key";

/** The admin key, kept for this browser tab only. */
export function useAdminKey(): [string | null, (key: string | null) => void] {
  const [key, setKey] = useState<string | null>(() => sessionStorage.getItem(STORE));
  const update = (next: string | null) => {
    if (next) sessionStorage.setItem(STORE, next);
    else sessionStorage.removeItem(STORE);
    setKey(next);
  };
  return [key, update];
}
