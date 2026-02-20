import { ResultsDashboard } from "../components/ResultsDashboard";
import { SiteFooter, SiteHeader } from "../components/SiteShell";

export default function ResultsPage() {
  return (
    <main>
      <SiteHeader />
      <ResultsDashboard />
      <SiteFooter />
    </main>
  );
}

