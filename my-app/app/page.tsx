import FlightForm from "@/components/FlightForm";
import Header from "@/components/Header";
import LLMBar from "@/components/ui/LLMBar";
import { Main } from "next/document";

export default function Home() {
  return (
    <main className=" h-full w-full">
      <Header />
      <FlightForm />
      <LLMBar />
    </main>
  );
}