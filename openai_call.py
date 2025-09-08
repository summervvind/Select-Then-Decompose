from openai import AsyncOpenAI, OpenAI
import asyncio

API_KEY = ''
BASE_URL = ''

async def _async_get_api_call(user_message, api_key, model, base_url, max_tokens, temperature, stop):
    """
    Call API and return result and token cost.
    """
    async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    response = await async_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": user_message},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
        stop=stop
    )

    usage = {
        "prompt_tokens": response.usage.prompt_tokens,  
        "completion_tokens": response.usage.completion_tokens, 
    }

    return response.choices[0].message.content, usage


async def get_openai_response(args, input, stop=None):
    """
    Call API and return result and token cost.
    """
    api_key = API_KEY
    base_url = BASE_URL
    model = args.model
    max_tokens = args.max_tokens
    temperature = args.temperature


    result, usage = await _async_get_api_call(input, api_key, model, base_url, max_tokens, temperature, stop)
    return result, usage