import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* 
   * Remove "export" to allow Next.js server features like Rewrites 
   * which we need to talk to the Python Backend 
   */
  images: {
    unoptimized: true,
  },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination:
          process.env.NODE_ENV === 'development'
            ? 'http://127.0.0.1:8000/:path*' // Local Python Backend
            : (process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000') + '/:path*', // Production Backend
      },
      {
        // Proxy root requests if you want /predict directly
        source: '/predict',
        destination:
          process.env.NODE_ENV === 'development'
            ? 'http://127.0.0.1:8000/predict'
            : (process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000') + '/predict',
      }
    ];
  },
};

export default nextConfig;
