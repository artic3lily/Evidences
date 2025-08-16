# async_example.py
# Asynchronous Programming practice for Week 1

import asyncio

async def greet_user(name):
    await asyncio.sleep(1)  # Simulate async I/O task (shortened for testing speed)
    return f"Hello, {name}!"

async def main():
    result = await asyncio.gather(
        greet_user("Suyasha"),
        greet_user("Fragger")
    )
    print(result)  # Output: ['Hello, Suyasha!', 'Hello, Fragger!']

if __name__ == "__main__":
    asyncio.run(main())
