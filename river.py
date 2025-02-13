from anthropic import AsyncAnthropic
import tenacity
import os
from typing import Literal
import asyncio

# Constants
RIDDLE = """
Answer the following riddle:

A farmer wants to cross a river and take with him a wolf, a goat and a cabbage. 
He has a boat with three secure separate compartments. If the wolf and the goat are alone on one shore, the wolf will eat the goat. 
If the goat and the cabbage are alone on the shore, the goat will eat the cabbage. 
How can the farmer efficiently bring the wolf, the goat and the cabbage across the river without anything being eaten?
"""

PromptType = Literal["raw", "cot", "rb"]


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
    retry=tenacity.retry_if_exception_type(Exception),
)
async def query_claude(client: AsyncAnthropic, messages: list) -> str:
    response = await client.messages.create(
        model="claude-3-5-sonnet-latest", max_tokens=4096, messages=messages
    )
    return response.content[0].text


async def run_experiment(
    prompt_type: PromptType, iteration: int, client: AsyncAnthropic
):
    # Construct the appropriate prompt based on type
    if prompt_type == "raw":
        messages = [{"role": "user", "content": RIDDLE}]
    elif prompt_type == "cot":
        messages = [{"role": "user", "content": f"{RIDDLE}\nThink step-by-step."}]
    else:  # rb case
        # First, get the repeat back
        messages = [
            {
                "role": "user",
                "content": f"{RIDDLE}\nBut before you do, please explain the task that I've asked you to perform to confirm your understanding.",
            }
        ]
        repeat_back = await query_claude(client, messages)
        # Then proceed
        messages = [
            {
                "role": "user",
                "content": f"{RIDDLE}\nBut before you do, please explain the task that I've asked you to perform to confirm your understanding.",
            },
            {"role": "assistant", "content": repeat_back},
            {"role": "user", "content": "Proceed."},
        ]

    # Get the response
    response = await query_claude(client, messages)

    # Save to file
    filename = f"prompt-{prompt_type}-result-{iteration}.txt"
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", filename), "w") as f:
        f.write(response)


async def main():
    client = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # Create all experiment tasks
    tasks = []
    for prompt_type in ["raw", "cot", "rb"]:
        for i in range(50):
            tasks.append(run_experiment(prompt_type, i, client))

    # Run all experiments in parallel
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
