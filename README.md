# Short-Video Comment AI Analysis Platform

## Product Positioning

This demo is an AI-assisted analysis platform for Bilibili short-video comments. It is designed for trend monitoring, early public-opinion screening, comment viewpoint synthesis, user feedback analysis, and course or thesis demonstrations.

The workflow follows a complete analysis chain: video -> comments -> sentiment -> topics -> summary -> report. It helps users quickly identify overall sentiment, group differences, regional differences, core issues, and potential discussion focus points from large volumes of short comments.

## Key AI Capabilities

### 1. LLM Event-Level Sentiment Analysis

The platform uses DeepSeek to classify comments into positive, neutral, and negative labels. Unlike generic sentiment analysis, event-level sentiment analysis judges the comment stance around the specific event, dispute, product, or public issue discussed in the video.

### 2. AI-Generated Event Prompt

When event-level sentiment analysis is selected, the platform can generate a prompt tailored to the current video context. It uses video metadata, uploader information, publish time, sample comments, and optional video summaries to identify the core event and produce a consistent classification prompt.

### 3. Subtitle Summary and ASR Transcription

The video summary module first tries to read Bilibili subtitles and generate the core content, key timeline, main viewpoints, and analysis context. If subtitles are unavailable or unstable, the platform can use OpenAI ASR to transcribe the video audio and use the transcript for summary generation.

### 4. AI Sentiment Difference Report

After sentiment analysis, the platform can generate a report based on overall sentiment distribution, gender differences, regional differences, and high-frequency words. Reports are generated only after the user clicks the button to avoid repeated API calls on page refresh.

### 5. AI Topic Analysis and Topic Naming

The topic analysis module uses BTM (Biterm Topic Model) for short-text topic modeling. DeepSeek can name each topic from its keywords and generate a topic analysis report using topic shares, representative comments, and optional sentiment cross results.

## Recommended Workflow

1. Open **Data Crawling**, enter a Bilibili video URL or BV ID, and set the maximum crawl pages.
2. Click the crawl button to collect top-level video comments.
3. Open **Data View** to check comment count, video information, gender distribution, IP location, likes, and comment content.
4. Open **Video Summary** to fetch subtitles or run ASR transcription, then generate a structured summary.
5. Open **Sentiment Analysis**, choose a prompt template, set analysis count and concurrency, and run DeepSeek classification.
6. Review overall distribution, gender differences, regional distribution, and the AI sentiment report.
7. Open **Topic Analysis** to run BTM topic modeling, view the word cloud, topic keywords, topic distribution, and AI topic report.
8. Download comment data, sentiment results, or topic files as needed.

## Configuration

Before first use, configure these items in **System Settings** in the sidebar:

- **Bilibili Cookie**: improves comment crawling stability. Missing or expired cookies may cause incomplete data or failed crawling.
- **DeepSeek API Key**: used for sentiment classification, AI prompt generation, sentiment reports, and topic reports.
- **OpenAI ASR API Key**: needed only when using ASR audio transcription in Video Summary.
- **Stopwords**: used for word-frequency statistics, topic modeling, and chart hover keywords.

## Scope and Risks

This is a demo analysis tool for exploration, presentation, and preliminary judgment. Analysis results should not be used as the sole basis for business decisions, public-opinion conclusions, user profiling, or high-risk decisions.

The platform depends on third-party interfaces and external model services. Bilibili APIs, subtitle availability, DeepSeek API behavior, and OpenAI ASR availability or pricing may change. Do not expose cookies or API keys in public environments.
