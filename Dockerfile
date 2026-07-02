FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir centering-lgram
RUN python -m spacy download en_core_web_sm

ENTRYPOINT ["centering-lgram"]
CMD ["--help"]
