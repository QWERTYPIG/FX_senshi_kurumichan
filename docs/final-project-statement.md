# Machine Learning (NTU, Fall 2026): Final Project

- Instructor: Hsuan-Tien Lin
- TA email: html_ta@csie.ntu.edu.tw
- Release date: 2026/10/24
- Report due: 2026/12/30 13:00
- Competition site: https://final.html2026.csie.org

Unless permission is granted by the instructor in advance, no late submissions will be allowed. That is, you will not be allowed to submit your report after the deadline. The gold medals cannot be used for the final project.

Any form of cheating, lying, or plagiarism will not be tolerated. Students can get zero scores and/or fail the class and/or be kicked out of school and/or receive other punishments for such misconduct.

You need to write your report in English or Chinese with the common math notations introduced in class or in the problems. We do not accept solutions written in any other language.

## Introduction

In this final project, you will act as a quantitative research team. Your goal is to predict the intraday direction of a fixed set of Taiwanese stocks from the provided historical market and global data. The project is structured as a competition consisting of a static phase and a dynamic phase. Together, they assess not only predictive performance but also experimental discipline, reproducibility, and the ability to maintain a model under a realistic submission schedule. The competition is hosted at https://final.html2026.csie.org.

## Data Sets

The training data set, `fin_pred_data.zip`, can be downloaded from the dashboard of the competition site. It contains 100 stock-level CSV files, a global-data CSV, and an information file describing the available series. You are responsible for examining the schema, handling missing values, preventing look-ahead leakage, and constructing a validation procedure that respects time order.

You may incorporate any publicly available features into both training and prediction if they can be reasonably reproduced. Please think of this as a company setting in which your colleagues will want to reproduce your solution. Private or paid features cannot be used unless they are shared with the whole class.

## Prediction Task and File Format

You are asked to make a prediction for a fixed set of 50 stocks. For stock $i$ on trading date $t$, let $o_{i,t}$ and $c_{i,t}$ denote the adjusted open and adjusted close prices. You must submit a binary prediction

$$\hat{y}_{i,t} \in \{0, 1\}, \qquad \hat{y}_{i,t} = 1 \text{ meaning “up” and } \hat{y}_{i,t} = 0 \text{ meaning “down.”}$$

The competition site checks the header, the complete list of 50 stock IDs, the number of rows and columns, and every prediction value. A file with missing, duplicated, reordered, or additional stock IDs will be rejected. You can use the phase-specific `sample_submission.csv` from the competition site as the template.

## Phase 1: Static Prediction

In Phase 1, your team must predict the direction of 50 stocks on 84 specified trading dates, from June 1 through September 30, 2026. Download the Phase 1 `sample_submission.csv`, keep the stock IDs and date headers unchanged, and replace every placeholder value with either 0 or 1. The completed file should contain 4200 predictions, one for every stock-date pair.

Upload the completed CSV before the Phase 1 deadline and confirm that it appears in Submission Status. Your team may submit multiple versions while the phase is open. Every accepted version is retained, and the Phase 1 leaderboard uses your team’s highest-scoring version. Therefore, a later submission with a lower score will not replace your earlier best result on the leaderboard.

The purpose of Phase 1 is to help you become familiar with the competition site and to let you repeatedly validate your solutions (usually called backtesting in computational finance). It will be a minor part of your score, and any team that performs sufficiently well (i.e., surpassing some TA baseline) will receive full credit for this part. While there is no official penalty even if you attempt to misuse the system (such as inferring the hidden labels strategically), you run a huge risk of doing worse in Phase 2 if you do not act professionally in Phase 1.

## Phase 2: Dynamic Prediction

Phase 2 is divided into separate daily sessions. For each session, download the `sample_submission.csv` of the day. The file contains exactly two columns: the stock IDs and the session date in YYYY-MM-DD format. Keep the header and all 50 stock IDs unchanged, enter one 0/1 prediction for each stock, and upload the completed CSV before that session’s deadline.

Your team may submit multiple versions while the session is open, but only the latest valid submission received before the deadline is counted. Please check the Submission Status after every upload. A record marked Pending indicates that the submission has been accepted and will receive a score later. Future sessions are not open for student submissions until their submission windows begin.

Phase 2 contains an ordinary stage and a competition stage. The ordinary stage (before November 15, 2026) is meant to familiarize you with the system and to establish your submission process, and your performance will not be part of your score. The competition stage (between November 16 and December 24) will be the most intense part of this final project, with cash prizes provided by Prof. Shou-de Lin (subject to eligibility rules), and your performance during this stage will be part of your score.

## Evaluation

Define the realized intraday return as

$$r_{i,t} = \frac{c_{i,t} - o_{i,t}}{o_{i,t}}.$$

The contribution of one prediction is $(2\hat{y}_{i,t} - 1)r_{i,t}$. Therefore, a correct direction receives a positive contribution equal to the magnitude of the return, while an incorrect direction receives the corresponding negative contribution. For a submission containing $N$ predictions, the score is

$$\text{Score} = \frac{1}{N}\sum_{(i,t)} (2\hat{y}_{i,t} - 1)r_{i,t}.$$

Here $N = 4200$ for Phase 1 and $N = 50$ for each Phase 2 session. Larger positive scores indicate better returns.

## Leaderboards and Submission Records

The website provides a Phase 1 leaderboard and other Phase 2 leaderboards with daily and cumulative views.

For ordinary Phase 2 sessions, the cumulative table can be viewed either by total score or by average score. The average is the total score divided by the number of submission days completed by the team.

For competition Phase 2 sessions, the leaderboard is handled differently. The scores count only competition days for which the administrator has uploaded the answer and completed the scoring, and the teams are ranked by total score only. On a competition day, if a team has not provided any valid submission before the deadline, the system creates an automatic submission that predicts 0 for every stock. This automatic submission applies only to sessions marked as competition days; it does not apply to ordinary sessions.

Every team member can review all current and historical uploads made by members of the same team in Submission Status. The page shows the uploader, the time, an optional note, the score or Pending state, and whether the record is current, historical, or system-generated. The original prediction CSV can be downloaded from each record.

## Competition Steps

1. Log in with the temporary password issued by an administrator, replace it when prompted, and verify your assigned team.
2. Download the training data and the appropriate `sample_submission.csv`. Do not alter stock IDs or headers.
3. Train and validate your model locally. Generate only 0/1 predictions.
4. Upload the CSV, optionally add a short note, and confirm that the record appears in Submission Status.
5. If the score is Pending, wait for the administrator to upload the answer. Recheck the status and leaderboard after scoring completes.

Please note that submissions belong to teams rather than individual team members, and any team member may upload a new version.

## Survey Report

You need to study at least FOUR machine learning approaches using the data. Then, you should compare those approaches from several different perspectives, such as (but not limited to) accuracy, stability across the two stages, efficiency, scalability, and interpretability. Finally, you need to select THE BEST ONE of those approaches as your final recommendation and provide the “pros and cons” of the choice. The report from the study is considered the most important part of the final project.

By “approach”, we mean a reproducible pipeline that involves machine learning. So in addition to learning models, any pre-processing and post-processing steps can be part of the approach. We will not be strict on what constitutes an “approach”, but as you can imagine, if your four approaches are overly similar, your evaluator (TAs, boss) could question whether your BEST recommendation is convincing.

The survey report should be no more than SEVEN A4 pages with readable font sizes. The most important criterion for evaluating your report is reproducibility. Thus, in addition to the outline above, you should also describe how you pre-process your data, such as the features you build; introduce the approaches you tried and provide specific references, especially for those approaches that we did not cover in class; list your experimental settings and the parameters you used (or chose) clearly. Other criteria for evaluating your survey report would include, but are not limited to, clarity, strength of your reasoning, “correctness” in using machine learning techniques, the workloads of team members, and proper citations.

Our sincere suggestion: Think of your TAs as your boss who wants to be convinced by your report.

For grading purposes, a minor but required part of your survey report for a two-, three-, or four-people team (see the rules below) is how you balance your workloads.

## Misc Rules

**Report:** Please upload one report per team electronically on Gradescope. You do not need to submit a hard copy. Your team name on your report must be the same as your team name on the competition site. The report is due at 13:00 on 2026/12/30. The gold medals cannot be used for the final project.

**Teams:** By default, you are asked to work as a team of size FOUR. A one-person, two-people or three-people team is allowed only if you are willing to be as good as a four-people team. It is expected that all team members share balanced workloads. Any form of unfairness, such as the intention to cover for other members or any unethical trading, is considered a violation of the honesty policy and will cause some or all members to receive lower, zero or even negative scores.

**Algorithms:** You can use any algorithms, regardless of whether they were taught in class.

**Packages:** You can use any software packages for the purpose of experiments, but please provide proper references in your report for reproducibility.

**Source Code:** You do not need to upload your source code for the final project. Nevertheless, please keep your source code until 2027/01/31 for the graders’ possible inspection or demo requests.

**Grade:** The final project is worth 1920 points. That is, it is equivalent to 8 usual homework sets, given our announced grading scheme. Your efforts/rank on Phase 1 submissions will be worth a total of 120 points; your efforts/rank on Phase 2 competition submissions will be worth a total of 360 points; your report will be worth 1440 points.

**Collaboration:** The general collaboration policy applies. Outside the competitions, we still encourage collaboration and discussion between different teams.

## Deadlines

- Team sign-up deadline: 2026/10/20 13:00 UTC+8
- Phase 1 submission deadline: 2026/12/24 23:59 UTC+8
- Phase 2 submission deadline: Each session closes at 09:00 UTC+8 on its corresponding trading date. Within Phase 2, the official competition days are November 16 to 20, November 23 to 27, November 30 to December 4, December 7 to 11, December 14 to 18, and December 21 to 24, 2026.
- Report due: 2026/12/30 13:00 UTC+8
- Competition workshop (required for those eligible for the competition prize; optional for others): TBA
- Source code retention: until 2027/01/31 23:59 UTC+8
