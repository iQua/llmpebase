"""
Getting the Claude model for inference.
either claude-v1.3 or claude-instant-v1.0.

There are two types of prompts for claude model.

Type 1:

    Human:
    xxx.

    Q: xxx?
    A: xxx.

    Q: xxx?
    A: xxx.

    Q: xxx?
    A: .

    anthropic.AI_PROMPT

Type 2:

    Human:
    xxx.
    anthropic.AI_PROMPT

    Human:
    Q: xxx?

    anthropic.AI_PROMPT
    A: xxx.
    
    Human:
    Q: xxx?

    anthropic.AI_PROMPT
    A: xxx.

    Human:
    Q: xxx?
    
    anthropic.AI_PROMPT
    A: .


    
In summary, 
anthropic.HUMAN_PROMPT and anthropic.AI_PROMPT can be 
added to each part.
or 
They are only added to the start and end of the prompt.

"""
import os
import anthropic

from dotenv import load_dotenv

# there must have a .env file containing keywords
# ANTHROPIC_KEY
load_dotenv()

ANTHROPIC_KEY = os.getenv("ANTHROPIC_KEY")


class ClaudeRequest(object):
    """A class to forward the Claude model for the response."""

    def get_response(self, **kwargs):
        """Getting one response from the model."""

        client = anthropic.Anthropic(api_key=ANTHROPIC_KEY)
        response = client.completions.create(**kwargs).completion
        return response

    def create_claude_prompt(self, prompt):
        """Creating the prompt available to claude model."""
        return anthropic.HUMAN_PROMPT + prompt + anthropic.AI_PROMPT

    def forward(
        self,
        prompt: str,
        model_name: str = "claude-instant-v1.0",
        temperature: float = 0.0,
        max_tokens_to_sample: int = 300,
    ):
        """Performing the request to get the response from the model."""

        claude_prompt = self.create_claude_prompt(prompt)
        # obtain Claude response
        model_response = self.get_response(
            model=model_name,
            prompt=claude_prompt,
            stop_sequences=[anthropic.HUMAN_PROMPT],
            max_tokens_to_sample=max_tokens_to_sample,
            temperature=temperature,
        )

        return model_response


if __name__ == "__main__":
    claude_api = ClaudeRequest()
    summary_prompt = """Here is the transcript of the video titled "Anthropic's new 100K context window model is insane!" by AssemblyAI:
    <transcript>
    And it worked and saved the file. So let's take a quick look. Here we get the text, 'The following is a conversation with John Carmack.' So it worked, and as you can see, this is a very long file here. So it would take us a lot of time to consume this content. So the first thing I want to do is totell Claude to give us a summary of this. So let's load our text again, and then first, let's get a rough estimation of how many words are in it by calling the dot split method and then also by applying our rule of thumb to get the tokens. So as you can see, we have almost 58,000 words in it and 77,000 tokens, so this should fit into the model.
    <transcript>

    You are an expert at writing factual summaries.

    write a summary of the transcript in an about 10 sentences.

    """
    claude_response = claude_api.forward(prompt=summary_prompt)
