"""Shared test fixtures for agent_core tests.

Currently provides a `mock_inference_server` fixture used by fetcher tests:
a tiny Starlette app served by uvicorn on a random local port, exposing the
HTML/edge-case routes the URLFetcher tests need.

Adapted from PAL's tests/conftest.py — trimmed to only the routes required
by agent_core's URL-fetcher tests (no pal-specific imports).
"""
import asyncio
import socket
from collections.abc import AsyncGenerator

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
import uvicorn


async def mock_page_html(request: Request):
    """Return a basic HTML page for fetcher tests."""
    return Response(
        "<html><head><title>Test Page</title></head>"
        "<body>"
        "<nav id=\"navigation\"><ul><li>Home</li><li>About</li><li>Nav junk</li></ul></nav>"
        "<main><article id=\"content\">"
        "<h1>Test Article</h1>"
        "<p>This is the main content. Extract me. This paragraph contains important information.</p>"
        "<p>Second paragraph with more main content for the article body extraction test.</p>"
        "<p>Third paragraph with additional content to ensure trafilatura picks this as main.</p>"
        "<p>Fourth paragraph confirming this is the primary content zone of the page.</p>"
        "</article></main>"
        "<footer id=\"footer\"><p>Footer junk copyright 2024</p></footer>"
        "</body></html>",
        media_type="text/html",
    )


async def mock_page_too_large(request: Request):
    """Return a response with a too-large Content-Length."""
    return Response(
        "tiny body",
        media_type="text/html",
        headers={"Content-Length": "999999999"},
    )


async def mock_page_binary(request: Request):
    """Return a binary content-type."""
    return Response(
        b"\x00\x01\x02\x03",
        media_type="application/octet-stream",
    )


async def mock_page_404(request: Request):
    return Response("not found", status_code=404)


async def mock_page_redirect(request: Request):
    return Response(
        "",
        status_code=302,
        headers={"location": "http://internal-service:9999/admin"},
    )


async def mock_page_no_content_type(request: Request):
    # Starlette sets a default content-type if we don't override - use raw 200
    return Response(
        "<html><body>no content-type</body></html>",
        media_type=None,
        headers={"content-type": ""},
    )


async def mock_page_with_code(request: Request):
    """Return an HTML page containing a code block."""
    return Response(
        "<html><head><title>Code Example</title></head>"
        "<body>"
        "<article>"
        "<h1>Code Tutorial</h1>"
        "<p>This tutorial shows a simple function. Here is example code for a greeting function.</p>"
        "<p>The function below demonstrates basic Python syntax and string formatting.</p>"
        "<pre><code>def hello(name):\n    return f\"Hello, {name}!\"\n\nprint(hello(\"world\"))</code></pre>"
        "<p>This function takes a name parameter and returns a formatted greeting string.</p>"
        "<p>You can call it with any name to get a personalized greeting message.</p>"
        "</article>"
        "</body></html>",
        media_type="text/html",
    )


mock_app = Starlette(routes=[
    Route("/page.html", mock_page_html, methods=["GET"]),
    Route("/too-large", mock_page_too_large, methods=["GET"]),
    Route("/binary", mock_page_binary, methods=["GET"]),
    Route("/missing", mock_page_404, methods=["GET"]),
    Route("/redirect", mock_page_redirect, methods=["GET"]),
    Route("/no-content-type", mock_page_no_content_type, methods=["GET"]),
    Route("/page-with-code.html", mock_page_with_code, methods=["GET"]),
])


@pytest.fixture()
async def mock_inference_server() -> AsyncGenerator[str, None]:
    """Start a mock HTTP server, yield its base URL."""
    # Find a free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    config = uvicorn.Config(mock_app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)

    task = asyncio.create_task(server.serve())
    # Wait for server to start
    while not server.started:
        await asyncio.sleep(0.01)

    yield f"http://127.0.0.1:{port}"

    server.should_exit = True
    await task
