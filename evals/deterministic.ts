import "dotenv/config";
import { v4 as uuidv4 } from "uuid";
import { evaluate, EvaluationResult } from "langsmith/evaluation";
import { graph } from "../src/agent/index";
import { Client, Example, Run } from "langsmith";
import { MemorySaver } from "@langchain/langgraph";

const listExamples = (datasetId: string) => {
  const lsClient = new Client();
  return lsClient.listExamples({ datasetId });
};

async function runGraph(inputs: Record<string, any>) {
  const checkpointer = new MemorySaver();
  // Ensure we interrupt after generating the artifact.
  graph.interruptAfter = ["updateArtifact"];
  graph.checkpointer = checkpointer;
  const config = {
    configurable: {
      thread_id: uuidv4(),
    },
  };
  // Set the state as the result of the first node.
  // This will cause the graph to start on the router,
  // which will route it to `updateArtifact`.
  await graph.updateState(
    config,
    {
      ...inputs,
      next: "updateArtifact",
    },
    "generatePath"
  );
  return graph.invoke(inputs, config);
}

function evaluateOutputs(run: Run, example?: Example): EvaluationResult {
  if (!example) {
    throw new Error("No example provided");
  }
  const { inputs: exampleInputs, outputs: exampleOutputs } = example;
  if (!exampleOutputs) {
    throw new Error("No outputs provided");
  }
  const { outputs: runOutputs } = run;
  if (!runOutputs) {
    throw new Error("No outputs provided");
  }
  const originalArtifactContent = exampleInputs.artifacts[0].content;
  const updatedArtifactContent = runOutputs.artifacts[0].content;
  const expectedChange = exampleOutputs.expectedGeneration;

  const originalArtifactContentStart = originalArtifactContent.slice(
    0,
    exampleInputs.highlighted.startCharIndex
  );
  const originalArtifactContentEnd = originalArtifactContent.slice(
    exampleInputs.highlighted.endCharIndex
  );
  const fullExpectedContent = `${originalArtifactContentStart}${expectedChange}${originalArtifactContentEnd}`;

  if (updatedArtifactContent !== fullExpectedContent) {
    return {
      key: "correct_generation",
      score: false,
    };
  } else {
    return {
      key: "correct_generation",
      score: true,
    };
  }
}

async function runEval() {
  const datasetId = "1a30e824-f13c-4b26-b45d-ab63f8e20f08";

  await evaluate(runGraph, {
    data: listExamples(datasetId),
    evaluators: [evaluateOutputs],
    experimentPrefix: "Highlight generation",
  });
}

runEval();
