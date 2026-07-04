import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /*
   * Remove "export" to allow Next.js server features like Rewrites
   * which we need to talk to the Python Backend
   */
  images: {
    unoptimized: true,
  },
  // Produces a minimal self-contained server bundle (.next/standalone) for
  // the frontend Docker image, instead of requiring a full node_modules
  // install in the runtime container.
  output: 'standalone',
  async rewrites() {
    // NEXT_PUBLIC_API_URL always wins when set (e.g. pointing at a specific
    // local port, a docker-compose service name, or a deployed backend).
    // Falls back to the conventional local FastAPI dev port otherwise.
    const backendUrl = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';
    return [
      {
        source: '/api/:path*',
        destination: `${backendUrl}/:path*`,
      },
      {
        // Proxy root requests if you want /predict directly
        source: '/predict',
        destination: `${backendUrl}/predict`,
      }
    ];
  },
};

export default nextConfig;
