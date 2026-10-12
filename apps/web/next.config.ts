import type { NextConfig } from "next";

// The GitHub Pages build (GITHUB_PAGES=true) is a static export served under
// /cardiovascular-risk. It has no server, so it calls the backend directly
// (see app/lib/apiUrl.ts). Other builds keep the server setup with rewrites.
const isPages = process.env.GITHUB_PAGES === "true";

const pagesConfig: NextConfig = {
  output: "export",
  basePath: "/cardiovascular-risk",
  trailingSlash: true,
  images: { unoptimized: true },
  env: {
    NEXT_PUBLIC_BASE_PATH: "/cardiovascular-risk",
    NEXT_PUBLIC_DIRECT_API: "true",
  },
};

const serverConfig: NextConfig = {
  images: {
    unoptimized: true,
  },
  // Produces a minimal self-contained server bundle (.next/standalone) for
  // the frontend Docker image, instead of requiring a full node_modules
  // install in the runtime container.
  output: "standalone",
  async rewrites() {
    // NEXT_PUBLIC_API_URL always wins when set (e.g. pointing at a specific
    // local port, a docker-compose service name, or a deployed backend).
    // Falls back to the conventional local FastAPI dev port otherwise.
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
      {
        // Proxy root requests if you want /predict directly
        source: "/predict",
        destination: `${backendUrl}/predict`,
      },
    ];
  },
};

const nextConfig: NextConfig = isPages ? pagesConfig : serverConfig;

export default nextConfig;
