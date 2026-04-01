const API_BASE_URL_KEY = 'ai-factory-api-base-url';
const DEFAULT_API_BASE = '/api/v1';

export function getApiBaseUrl(): string {
  return localStorage.getItem(API_BASE_URL_KEY) || DEFAULT_API_BASE;
}

export function setApiBaseUrl(url: string): void {
  const sanitized = url.replace(/\/+$/, '');
  localStorage.setItem(API_BASE_URL_KEY, sanitized);
}

export async function apiRequest<T = unknown>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const base = getApiBaseUrl();
  const url = `${base}${endpoint}`;
  const res = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new Error(`API error ${res.status}: ${text}`);
  }

  const contentType = res.headers.get('content-type');
  if (contentType?.includes('application/json')) {
    return res.json();
  }
  return res.text() as unknown as T;
}

export function defaultQueryFn({ queryKey }: { queryKey: readonly unknown[] }): Promise<unknown> {
  const endpoint = queryKey[0] as string;
  const id = queryKey[1];
  const suffix = queryKey[2];
  let path = endpoint;
  if (id !== undefined) {
    path = `${endpoint}/${id}`;
  }
  if (suffix !== undefined) {
    path = `${path}/${suffix}`;
  }
  return apiRequest(path);
}
