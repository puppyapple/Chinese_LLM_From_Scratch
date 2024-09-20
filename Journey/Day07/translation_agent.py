import asyncio
from typing import List, Union
from openai import AsyncAzureOpenAI
import tiktoken
from icecream import ic
from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure_configs import API_KEY, ENDPOINT, MODEL

# 创建异步客户端
async_client = AsyncAzureOpenAI(
    api_key=API_KEY,
    api_version="2024-02-01",
    azure_endpoint=ENDPOINT,
)

MAX_TOKENS_PER_CHUNK = 1000


async def get_completion(
    prompt: str,
    system_message: str = "You are a helpful assistant.",
    model: str = MODEL,
    temperature: float = 0.3,
    json_mode: bool = False,
) -> Union[str, dict]:
    if json_mode:
        response = await async_client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    else:
        response = await async_client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=1,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


async def one_chunk_initial_translation(
    source_lang: str, target_lang: str, source_text: str
) -> str:
    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    translation_prompt = f"""This is an {source_lang} to {target_lang} translation, please provide the {target_lang} translation for this text. \
Do not provide any explanations or text apart from the translation.
{source_lang}: {source_text}

{target_lang}:"""

    translation = await get_completion(
        translation_prompt, system_message=system_message
    )

    return translation


async def one_chunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    country: str = "",
) -> str:
    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions to improve the translation. \
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:
        reflection_prompt = f"""Your task is to carefully read a source text and a translation from {source_lang} to {target_lang}, and then give constructive criticisms and helpful suggestions to improve the translation. \

The source text and initial translation, delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT> and <TRANSLATION></TRANSLATION>, are as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's \n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    reflection = await get_completion(reflection_prompt, system_message=system_message)
    return reflection


async def one_chunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text: str,
    translation_1: str,
    reflection: str,
) -> str:
    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."

    prompt = f"""Your task is to carefully read, then edit, a translation from {source_lang} to {target_lang}, taking into
account a list of expert suggestions and constructive criticisms.

The source text, the initial translation, and the expert linguist suggestions are delimited by XML tags <SOURCE_TEXT></SOURCE_TEXT>, <TRANSLATION></TRANSLATION> and <EXPERT_SUGGESTIONS></EXPERT_SUGGESTIONS> \
as follows:

<SOURCE_TEXT>
{source_text}
</SOURCE_TEXT>

<TRANSLATION>
{translation_1}
</TRANSLATION>

<EXPERT_SUGGESTIONS>
{reflection}
</EXPERT_SUGGESTIONS>

Please take into account the expert suggestions when editing the translation. Edit the translation by ensuring:

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation and nothing else."""

    translation_2 = await get_completion(prompt, system_message)

    return translation_2


async def one_chunk_translate_text(
    source_lang: str, target_lang: str, source_text: str, country: str = ""
) -> str:
    translation_1 = await one_chunk_initial_translation(
        source_lang, target_lang, source_text
    )

    reflection = await one_chunk_reflect_on_translation(
        source_lang, target_lang, source_text, translation_1, country
    )
    translation_2 = await one_chunk_improve_translation(
        source_lang, target_lang, source_text, translation_1, reflection
    )

    return translation_2


def num_tokens_in_string(input_str: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(input_str))
    return num_tokens


async def multichunk_initial_translation(
    source_lang: str, target_lang: str, source_text_chunks: List[str]
) -> List[str]:
    system_message = f"You are an expert linguist, specializing in translation from {source_lang} to {target_lang}."

    translation_prompt = """Your task is to provide a professional translation from {source_lang} to {target_lang} of PART of a text.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>. Translate only the part within the source text
delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS>. You can use the rest of the source text as context, but do not translate any
of the other text. Do not output anything other than the translation of the indicated part of the text.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, you should translate only this part of the text, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

Output only the translation of the portion you are asked to translate, and nothing else.
"""

    translation_chunks = []
    for i in range(len(source_text_chunks)):
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = translation_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
        )

        translation = await get_completion(prompt, system_message=system_message)
        translation_chunks.append(translation)

    return translation_chunks


async def multichunk_reflect_on_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    country: str = "",
) -> List[str]:
    system_message = f"You are an expert linguist specializing in translation from {source_lang} to {target_lang}. \
You will be provided with a source text and its translation and your goal is to improve the translation."

    if country != "":
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.
The final style and tone of the translation should match the style of {target_lang} colloquially spoken in {country}.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    else:
        reflection_prompt = """Your task is to carefully read a source text and part of a translation of that text from {source_lang} to {target_lang}, and then give constructive criticism and helpful suggestions for improving the translation.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context for critiquing the translated part.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

When writing suggestions, pay attention to whether there are ways to improve the translation's:\n\
(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),\n\
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules, and ensuring there are no unnecessary repetitions),\n\
(iii) style (by ensuring the translations reflect the style of the source text and take into account any cultural context),\n\
(iv) terminology (by ensuring terminology use is consistent and reflects the source text domain; and by only ensuring you use equivalent idioms {target_lang}).\n\

Write a list of specific, helpful and constructive suggestions for improving the translation.
Each suggestion should address one specific part of the translation.
Output only the suggestions and nothing else."""

    reflection_chunks = []
    for i in range(len(source_text_chunks)):
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )
        if country != "":
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
                country=country,
            )
        else:
            prompt = reflection_prompt.format(
                source_lang=source_lang,
                target_lang=target_lang,
                tagged_text=tagged_text,
                chunk_to_translate=source_text_chunks[i],
                translation_1_chunk=translation_1_chunks[i],
            )

        reflection = await get_completion(prompt, system_message=system_message)
        reflection_chunks.append(reflection)

    return reflection_chunks


async def multichunk_improve_translation(
    source_lang: str,
    target_lang: str,
    source_text_chunks: List[str],
    translation_1_chunks: List[str],
    reflection_chunks: List[str],
) -> List[str]:
    system_message = f"You are an expert linguist, specializing in translation editing from {source_lang} to {target_lang}."

    improvement_prompt = """Your task is to carefully read, then improve, a translation from {source_lang} to {target_lang}, taking into
account a set of expert suggestions and constructive criticisms. Below, the source text, initial translation, and expert suggestions are provided.

The source text is below, delimited by XML tags <SOURCE_TEXT> and </SOURCE_TEXT>, and the part that has been translated
is delimited by <TRANSLATE_THIS> and </TRANSLATE_THIS> within the source text. You can use the rest of the source text
as context, but need to provide a translation only of the part indicated by <TRANSLATE_THIS> and </TRANSLATE_THIS>.

<SOURCE_TEXT>
{tagged_text}
</SOURCE_TEXT>

To reiterate, only part of the text is being translated, shown here again between <TRANSLATE_THIS> and </TRANSLATE_THIS>:
<TRANSLATE_THIS>
{chunk_to_translate}
</TRANSLATE_THIS>

The translation of the indicated part, delimited below by <TRANSLATION> and </TRANSLATION>, is as follows:
<TRANSLATION>
{translation_1_chunk}
</TRANSLATION>

The expert translations of the indicated part, delimited below by <EXPERT_SUGGESTIONS> and </EXPERT_SUGGESTIONS>, are as follows:
<EXPERT_SUGGESTIONS>
{reflection_chunk}
</EXPERT_SUGGESTIONS>

Taking into account the expert suggestions rewrite the translation to improve it, paying attention
to whether there are ways to improve the translation's

(i) accuracy (by correcting errors of addition, mistranslation, omission, or untranslated text),
(ii) fluency (by applying {target_lang} grammar, spelling and punctuation rules and ensuring there are no unnecessary repetitions), \
(iii) style (by ensuring the translations reflect the style of the source text)
(iv) terminology (inappropriate for context, inconsistent use), or
(v) other errors.

Output only the new translation of the indicated part and nothing else."""

    translation_2_chunks = []
    for i in range(len(source_text_chunks)):
        tagged_text = (
            "".join(source_text_chunks[0:i])
            + "<TRANSLATE_THIS>"
            + source_text_chunks[i]
            + "</TRANSLATE_THIS>"
            + "".join(source_text_chunks[i + 1 :])
        )

        prompt = improvement_prompt.format(
            source_lang=source_lang,
            target_lang=target_lang,
            tagged_text=tagged_text,
            chunk_to_translate=source_text_chunks[i],
            translation_1_chunk=translation_1_chunks[i],
            reflection_chunk=reflection_chunks[i],
        )

        translation_2 = await get_completion(prompt, system_message=system_message)
        translation_2_chunks.append(translation_2)

    return translation_2_chunks


async def multichunk_translation(
    source_lang, target_lang, source_text_chunks, country: str = ""
):
    translation_1_chunks = await multichunk_initial_translation(
        source_lang, target_lang, source_text_chunks
    )

    reflection_chunks = await multichunk_reflect_on_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        country,
    )

    translation_2_chunks = await multichunk_improve_translation(
        source_lang,
        target_lang,
        source_text_chunks,
        translation_1_chunks,
        reflection_chunks,
    )

    return translation_2_chunks


def calculate_chunk_size(token_count: int, token_limit: int) -> int:
    if token_count <= token_limit:
        return token_count

    num_chunks = (token_count + token_limit - 1) // token_limit
    chunk_size = token_count // num_chunks

    remaining_tokens = token_count % token_limit
    if remaining_tokens > 0:
        chunk_size += remaining_tokens // num_chunks

    return chunk_size


async def translate(
    source_lang,
    target_lang,
    source_text,
    country,
    max_tokens=MAX_TOKENS_PER_CHUNK,
):
    num_tokens_in_text = num_tokens_in_string(source_text)

    ic(num_tokens_in_text)

    if num_tokens_in_text < max_tokens:
        ic("Translating text as a single chunk")

        final_translation = await one_chunk_translate_text(
            source_lang, target_lang, source_text, country
        )

        return final_translation

    else:
        ic("Translating text as multiple chunks")

        token_size = calculate_chunk_size(
            token_count=num_tokens_in_text, token_limit=max_tokens
        )

        ic(token_size)

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4",
            chunk_size=token_size,
            chunk_overlap=0,
        )

        source_text_chunks = text_splitter.split_text(source_text)

        translation_2_chunks = await multichunk_translation(
            source_lang, target_lang, source_text_chunks, country
        )

        return "".join(translation_2_chunks)


async def main():
    text = """
    Random sentence: They are very excited and want to fly too.
    Features: Dialogue
    Summary: Tom and Anna are excited to go on a holiday with their parents, and they fly on a big plane to a place with sun and sand.
    Story: 
    Tom and Anna are brother and sister. They like to play with their toys and read books. They are very happy because they are going on a holiday with their mum and dad. They will fly on a big plane to a place with a lot of sun and sand.
    The day of the holiday comes and they pack their bags. They go to the airport and wait for their plane. They see many other planes flying in the sky. They are very excited and want to fly too.
    "Look, Anna, that plane is so big and fast!" Tom says.
    "Yes, Tom, and it has wings and a tail. I wonder where it is going," Anna says.
    They hear their mum call them. "Come on, kids, it's time to board our plane. We have to show our tickets and go through the gate."
    They follow their mum and dad and get on their plane. They find their seats and buckle their belts. They look out the window and see the ground and the cars and the people. They hear the pilot say something on the speaker.
    "Hello, everyone, this is your pilot speaking. Welcome aboard flight 123 to Sunny Beach. We are ready to take off. Please sit back and enjoy the flight."
    The plane starts to move and makes a loud noise. Tom and Anna feel the plane go faster and faster. They see the ground get smaller and smaller. They see the clouds get closer and closer. They are flying!
    "Wow, Anna, we are flying! We are in the sky!" Tom says.
    "I know, Tom, it's amazing! We are so high! Look, there is the sun!" Anna says.
    They smile and laugh and clap their hands. They are not sad at all. They are very happy. They are flying to their holiday.
"""
    result = await translate(
        source_lang="English",
        target_lang="Chinese",
        source_text=text,
        country="China",
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
