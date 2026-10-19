# Gold set: labelling guide

`to_label.csv` holds every embedded article published on **21–22 August 2026**: 282
articles from 18 outlets. These two days were chosen because the pipeline was running every
hour, every article has a real publish date, and the size fits the 200–300 the evaluation
plan called for.

**Do not open `system_snapshot.json` before you finish.** It holds the system's own answers;
seeing them would bias the labels toward the system.

## What to do

Fill the `event` column for every row. Give articles that report **the same event** the same
short name, and every other article a different name. The names themselves don't matter, only
which rows share one, so use anything readable: `imf-review-approval`, `kandy-landslide`.
Capitals and extra spaces are ignored.

You can sort and filter the sheet any way you like in Excel. Scoring matches rows by
`article_id`, so row order doesn't matter. Keep the file as CSV when you save it.

## The rule: what counts as the same event

An event is **one specific real-world occurrence**: one incident, one announcement, one
decision, one meeting, one statement, or one release of figures.

1. **Same occurrence, same event.** Articles from different outlets, or from the same outlet,
   that report the same occurrence get the same name, even if the headlines are worded
   differently.
2. **A new development is a new event.** A later arrest, a court hearing, a reply from another
   party, or a new decision about the same story is its own event, even though it belongs to
   the same running story.
3. **Recurring items are separate per occurrence.** Today's weather forecast, today's exchange
   rates and today's stock market close are each one event per day. Two outlets reporting the
   same day's figures share that day's event.
4. **Opinion and analysis.** A column or analysis piece that reports no specific new occurrence
   gets its own name.
5. **When unsure, give it its own name.** This makes any doubt fall toward splitting events,
   which is the safer error. Note any row you were unsure about in a separate list, so it can be
   reported.

## Scoring

```bash
venv/Scripts/python.exe gold_set.py score gold/labelled.csv
```

Save your finished sheet as `gold/labelled.csv` first. The output compares the deployed system
with plain HDBSCAN (no refinement), TF-IDF with K-Means, and every-article-alone, and writes
`gold/results.json`.

**Read ARI and pairwise F1 first.** Most events in this set have one article, and with that many
single-article events NMI, homogeneity and V-measure reward splitting: putting every article
alone scores a high NMI while finding no event at all.

A second labeller can label the same file independently. Pass their file with
`--second other.csv` to report how well the two labellers agree.
