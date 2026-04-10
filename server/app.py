from ecommerce_env import app  # noqa: F401


def main():
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
