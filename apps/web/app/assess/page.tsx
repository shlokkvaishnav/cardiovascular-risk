import { AssessmentWizard } from "../components/AssessmentWizard";
import { SiteFooter, SiteHeader } from "../components/SiteShell";

export default function AssessPage() {
  return (
    <main>
      <SiteHeader />
      <AssessmentWizard />
      <SiteFooter />
    </main>
  );
}

