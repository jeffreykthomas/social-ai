"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling the local reasoning model.
"""
import json
import random
import time 
import hashlib
import os

import sys

sys.path.append('../../')

from config.config_loader import get_config
from llm_client import LLMConfig, get_client

_config = get_config()
_llm_params = _config.get_llm_params()
if not _llm_params.get('model_name'):
  raise ValueError("llm.model_name must be configured in need_config.yaml")

_llm_config = LLMConfig(
  model_name=_llm_params['model_name'],
  device=_llm_params.get('device'),
  dtype=_llm_params.get('dtype', 'bfloat16'),
  max_new_tokens=_llm_params.get('max_new_tokens', 256),
  temperature=_llm_params.get('temperature', 0.7),
  top_p=_llm_params.get('top_p', 0.9),
  repetition_penalty=_llm_params.get('repetition_penalty', 1.05),
)


def _get_client():
  return get_client(_llm_config)


# Optional OpenAI client for legacy GPT_request and embeddings
try:
  from openai import OpenAI  # type: ignore
  _client = OpenAI()  # uses OPENAI_API_KEY env var
except Exception:  # pragma: no cover
  _client = None

def temp_sleep(seconds=0.1):
  time.sleep(seconds)

def ChatGPT_single_request(prompt): 
  temp_sleep()
  return _get_client().generate([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
  ])


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  try:
    return _get_client().generate([
      {"role": "system", "content": "You are a thoughtful assistant."},
      {"role": "user", "content": prompt},
    ])
  except Exception as exc:
    print ("ChatGPT ERROR", exc)
    return "ChatGPT ERROR"


def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  try:
    return _get_client().generate([
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": prompt},
    ])
  except Exception as exc:
    print ("ChatGPT ERROR", exc)
    return "ChatGPT ERROR"


def GPT4_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return fail_safe_response


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      # print ("---ashdfaf")
      # print (curr_gpt_response)
      # print ("000asdfhia")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return fail_safe_response


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose: 
        print (f"---- repeat count: {i}")
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass
  print ("FAIL SAFE TRIGGERED") 
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  temp_sleep()

  # Legacy compatibility layer:
  # The original Reverie code used the pre-v1 OpenAI SDK Completion API with engines like
  # "text-davinci-003". This repo now uses OpenAI SDK v1+ (`OpenAI()`), so we route those
  # legacy "completion" calls through chat completions.
  #
  # You can override the model used for these calls with:
  #   LEGACY_COMPLETION_MODEL=gpt-4o-mini   (default)
  #   LEGACY_COMPLETION_MODEL=gpt-4o
  #
  # If OpenAI isn't configured, we return a short error string and let callers fall back.

  if _client is None:
    return "LLM_UNAVAILABLE"

  engine = str(gpt_parameter.get("engine", "") or "")
  legacy_default = os.environ.get("LEGACY_COMPLETION_MODEL", "gpt-4o-mini")

  # Map common legacy engines to modern chat models
  legacy_map = {
    "text-davinci-003": legacy_default,
    "text-davinci-002": legacy_default,
    "davinci": legacy_default,
    "gpt-3.5-turbo": legacy_default,
  }
  model = legacy_map.get(engine, engine or legacy_default)

  # Map legacy param names to chat-completions equivalents (best-effort)
  try:
    temperature = float(gpt_parameter.get("temperature", 0.7))
  except Exception:
    temperature = 0.7
  try:
    max_tokens = int(gpt_parameter.get("max_tokens", 256))
  except Exception:
    max_tokens = 256

  stop = gpt_parameter.get("stop", None)
  if stop is not None and not isinstance(stop, (list, str)):
    stop = None

  try:
    resp = _client.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      temperature=temperature,
      max_tokens=max_tokens,
      stop=stop,
    )
    # openai>=1: resp.choices[0].message.content
    return (resp.choices[0].message.content or "").strip()
  except Exception:
    # Do not emit "TOKEN LIMIT EXCEEDED" sentinel strings; they break downstream parsers.
    return "LLM_ERROR"


def generate_prompt(curr_input, prompt_lib_file): 
  """
  Takes in the current input (e.g. comment that you want to classifiy) and 
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this 
  function replaces this substr with the actual curr_input to produce the 
  final promopt that will be sent to the GPT3 server. 
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file. 
  RETURNS: 
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  if type(curr_input) == type("string"): 
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):   
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt: 
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if verbose: 
    print (prompt)

  for i in range(repeat): 
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    try:
      ok = func_validate(curr_gpt_response, prompt=prompt)
    except Exception:
      ok = False
    if ok:
      try:
        return func_clean_up(curr_gpt_response, prompt=prompt)
      except Exception:
        # Treat clean-up failures as invalid and retry (or fall back).
        ok = False
    if verbose: 
      print ("---- repeat count: ", i, curr_gpt_response)
      print (curr_gpt_response)
      print ("~~~~")
  return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  # Backward/forward compatible embedding call:
  # - Old SDK: openai.Embedding.create(...)
  # - New SDK: OpenAI().embeddings.create(...)
  # This codebase uses the new SDK for chat (`_client.responses.create`) but the
  # original embedding call was left behind.
  #
  # We also provide a deterministic fallback embedding so classic simulations
  # can still run in environments without OpenAI configured (retrieval quality
  # will be reduced but the sim won't crash).
  #
  # Map deprecated model names to modern equivalents.
  # Allow global override via env.
  embed_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "").strip() or model
  if embed_model == "text-embedding-ada-002":
    embed_model = "text-embedding-3-small"

  # Preferred: new client (OpenAI SDK v1+)
  if _client is not None:
    try:
      resp = _client.embeddings.create(model=embed_model, input=[text])
      # openai>=1: resp.data[0].embedding
      emb = resp.data[0].embedding  # type: ignore[attr-defined]
      if isinstance(emb, list) and emb:
        return emb
    except Exception:
      pass

  # Secondary: attempt old SDK if installed (best-effort)
  try:
    import openai as _openai  # type: ignore
    try:
      _openai.api_key = openai_api_key
    except Exception:
      pass
    resp = _openai.Embedding.create(input=[text], model=model)
    return resp["data"][0]["embedding"]
  except Exception:
    pass

  # Fallback: deterministic pseudo-embedding (1536 dims; ada-002 compatible)
  # Note: this is not semantically meaningful, but keeps the pipeline alive.
  dims = 1536
  seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
  rng = random.Random(seed)
  return [rng.uniform(-1.0, 1.0) for _ in range(dims)]


if __name__ == '__main__':
  gpt_parameter = {"engine": "gpt-4o-mini", "max_tokens": 50, 
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0, 
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1: 
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt, 
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print (output)



















