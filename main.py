from src.workflow import Workflow
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    print("Hello from cert-agent!")

    workflow = Workflow()
    await workflow.run(exam_images=[])


if __name__ == "__main__":
    asyncio.run(main())
