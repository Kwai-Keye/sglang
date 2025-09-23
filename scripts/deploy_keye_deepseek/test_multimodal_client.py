#!/usr/bin/env python3
"""
Usage:
    python test_multimodal_client.py
"""

import json

import requests


class VLMClient:
    def __init__(self, base_url="http://localhost:30000"):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}

    def generate(self, messages, **kwargs):
        """Send generation request"""
        payload = {
            "model": "",
            "messages": messages,
            "n": 1,
            "temperature": 0.0,
            "max_tokens": 256,
            "top_k": 1,
            "ignore_eos": False,
            "skip_special_tokens": True,
            **kwargs,
        }
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self.headers,
            data=json.dumps(payload),
            timeout=1800,
        )
        response.raise_for_status()
        return response.json()

    def _create_message(self, media_type, media_path, text):
        """Create message with media content"""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": f"{media_type}_url",
                        f"{media_type}_url": {"url": media_path},
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]

    def image(self, image_path, text="Describe the image in detail."):
        """Generate response for image"""
        messages = self._create_message("image", image_path, text)
        return self.generate(messages)

    def video(self, video_path, text="Describe the video in detail."):
        """Generate response for video"""
        messages = self._create_message("video", video_path, text)
        return self.generate(messages)


def main():
    client = VLMClient()

    # Test image
    print("Test image request...")
    image_path = (
        "https://raw.githubusercontent.com/sgl-project/sglang/main/assets/logo.png"
    )
    response = client.image(image_path)
    print(response["choices"][0]["message"]["content"])
    print()

    # Test video
    print("Test video request...")
    video_path = "https://raw.githubusercontent.com/EvolvingLMMs-Lab/sglang/dev/onevision_local/assets/jobs.mp4"
    response = client.video(video_path)
    print(response["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
