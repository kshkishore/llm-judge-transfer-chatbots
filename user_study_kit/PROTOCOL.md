# User study kit: Do users agree with the LLM judges?

This kit lets you run the user study that the IJHCI manuscript names as its main next step. Its results could be added to the manuscript as a new section.

**The kit contains no participant data. Do not report any results until you have run the study.**

## Aim

Test which judge's conclusions match the judgments of people, and whether the rule-based checks and text metrics do. The study answers four questions:

- **U1. Agreement with each judge.** Do participants' ratings of chatbot replies agree more with J1 (Qwen3-VL-4B) or with J2 (Phi-4-mini)?
- **U2. Which chatbot is better.** Do participants judge G1 (Qwen3-VL-2B) better than G2 (SmolVLM2-2.2B), as J1 did, or the two alike, as J2 did?
- **U3. Format.** Do replies that break the requested format (the format rule) receive lower ratings for format?
- **U4. Misreadings.** For replies that J1 alone flags as misreadings, do participants agree that the reply misunderstands the user?

## Materials

The stimuli are 80 chatbot replies, in `stimuli.csv`:

- **Requests:** 40 requests, four per category, sampled at random with seed 20261121, across all ten categories.
- **Replies:** each request has the replies of both chatbots.
- **Order:** the rows are already shuffled.
- **Chart items:** chart items have an image in `images/`.
- **Blinding:** participants see only the request (with the chart) and the reply. They are not told which chatbot wrote it, and they never see the judges' scores.

`key.csv` maps each stimulus to its item, category and chatbot, the judges' scores, the rule results and selected metrics. **Never show it to participants.**

## Participants

- **Number:** 20–30 adults who are fluent readers of English. A total of 20 raters × 80 replies gives 1,600 ratings, enough to estimate per-reply means and inter-rater reliability.
- **Optional subgroup:** include some Indian English users, because some requests concern Indian English and Indian social conventions.
- **Time:** about 45–60 minutes.
- **Ethics and consent:** obtain approval from your institution's ethics committee before recruiting, and record informed consent. The materials contain no personal data.

## Procedure

Each participant is shown one item at a time:

1. **The request.** Shown as written by a user, with the chart if one is attached.
2. **The reply.** Shown as the chatbot's answer.

The participant then answers five rating questions (see `QUESTIONS.md`):

| Question | Scale | Corresponds to |
|---|---|---|
| Q1 Understanding | 1–5 | the judges' Objective dimension |
| Q2 Tone and register | 1–5 | Style and Tone |
| Q3 Form and length | 1–5 | Response format |
| Q4 Overall appropriateness | 1–7 | the overall score |
| Q5 Would you send or use this reply as it is? | yes / no | — |

Then present the items in a new random order for each participant if the survey tool allows; otherwise use two counterbalanced orders (forward and reversed). Include two attention checks, using clearly absurd replies that are not in `stimuli.csv`.

## Analysis (preregister before data collection)

`analyze_user_study.py` takes a long-format CSV with columns `participant, stimulus_id, q1_understanding, q2_tone, q3_form, q4_overall, q5_use`. It computes:

1. **Inter-rater reliability.** Krippendorff's α across participants, separately for each question.
2. **Agreement with each judge (U1).** It averages each reply's ratings across participants, then computes Spearman correlations of those means with J1's and J2's overall scores, with bootstrap confidence intervals. Q1 is also compared with the judges' Objective scores. The difference between the J1 and J2 correlations is tested with a bootstrap over replies.
3. **Chatbot comparison (U2).** A Wilcoxon signed-rank test of G1 vs. G2 on mean Q4, paired by request.
4. **Format rule (U3).** A Mann–Whitney test and AUC of Q3 for replies that pass vs. fail the format rule.
5. **Misreading flags (U4).** Mean Q1 for replies flagged by both judges, by J1 only, and by neither.
6. **Metrics.** Spearman correlations of mean Q4 with BLEU, BERTScore, entailment and length.

## Hypotheses to preregister

- **H-U1:** participants' mean overall ratings correlate more strongly with J1 than with J2.
- **H-U2:** participants rate G1 higher than G2.
- **H-U3:** replies that break the format rule receive lower Q3 ratings.
- **H-U4:** replies flagged by J1 only receive lower Q1 ratings than replies flagged by neither judge.
