/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  typedRoutes: true,
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: 'http://localhost:8001/api/v1/:path*',
      },
      {
        source: '/v1/:path*',
        destination: 'http://localhost:8001/v1/:path*',
      },
    ];
  },
};

export default nextConfig;
