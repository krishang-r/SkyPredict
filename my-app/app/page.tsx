import FlightForm from "@/components/FlightForm";
import Header from "@/components/Header";
import LLMBar from "@/components/ui/LLMBar";
import { Main } from "next/document";

export default function Home() {
  return (
    // added pb-24 (padding-bottom). change to pb-32 / pb-16 as needed
    <main className="h-full w-full pb-24">
      <Header />
      <FlightForm />
      <LLMBar />
    </main>
  );
}