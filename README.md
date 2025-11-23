# MLIA Final Project \- Team Organization

**Project:** Cardiac CMR Segmentation \+ Sequencer Regression (T4 \- Project \#8)  
**Team:** Victoria, Xiaoyu, Matthew, Lawson  
**Key Dates:** Presentation (Dec 1\) | Paper \+ Code (Dec 12\)

---

## TEAM ROLES

### Victoria \- Paper & Literature Lead

**Focus:** Lead all writing and literature review  
**Key Tasks:**

- Read & summarize Sequencer paper \+ related work  
- Create intro \+ lit review slides (2-3 slides)  
- Write paper intro, related work, background (\~3 pages)  
- Coordinate LaTeX platform setup

**Deadlines:** Slides (Nov 30\) | Paper sections (Dec 10\)

---

### Xiaoyu \- Segmentation (U-Net) Lead

**Focus:** Own entire segmentation pipeline  
**Key Tasks:**

- Download dataset and setup preprocessing  
- Train U-Net for heart contour segmentation  
- Generate all segmentation masks (⚠️ **CRITICAL PATH** \- blocks Matthew & Lawson)  
- Create segmentation methodology slides (1-2 slides)  
- Write segmentation paper section (\~1.5 pages)

**Deadlines:** U-Net trained \+ all masks ready (Nov 30\) | Slides (Nov 30\) | Paper section (Dec 10\)

---

### Matthew \- Feature Processing Lead

**Focus:** Bridge segmentation to regression  
**Key Tasks:**

- Design feature extraction from masks (contours? shape descriptors? time series?)  
- Implement feature extraction pipeline  
- Build baseline regression models for comparison  
- Create feature engineering slides (1-2 slides)  
- Write feature extraction paper section (\~1.5 pages)

**Deadlines:** Feature pipeline (Dec 3\) | Slides (Nov 30\) | Paper section (Dec 10\)  
**Dependencies:** Needs Xiaoyu's masks (Nov 30\)

---

### Lawson \- Sequencer Implementation & Integration Lead

**Focus:** Implement Sequencer and integrate full system  
**Key Tasks:**

- Adapt Sequencer from classification → regression  
- Set up GitHub repo and code structure  
- Integrate full pipeline (segmentation → features → Sequencer)  
- Run experiments, tune hyperparameters, collect results  
- Create Sequencer \+ results slides (1-2 slides)  
- Write experiments, results, discussion (\~2 pages)  
- Ensure final code runs end-to-end

**Deadlines:** Repo setup (Nov 25\) | Integration (Dec 5\) | Results (Dec 8\) | Slides (Nov 30\) | Paper sections (Dec 10\)  
**Dependencies:** Needs Xiaoyu's masks (Nov 30\) \+ Matthew's features (Dec 3\)

---

## HIGH-LEVEL TIMELINE

PHASE 1: SETUP (Nov 23-27)

├─ ALL: Read Sequencer paper

├─ Xiaoyu: Download dataset, start U-Net

├─ Victoria: Literature research

├─ Matthew: Design feature extraction

└─ Lawson: Study Sequencer, setup GitHub

PHASE 2: CORE WORK (Nov 27 \- Dec 1\)

├─ Xiaoyu: Train U-Net → generate masks (DONE BY NOV 30\)

├─ Matthew: Implement features (needs masks)

├─ Lawson: Code Sequencer architecture

└─ Victoria: Create lit review slides

PRESENTATION PREP (Nov 28 \- Dec 1\)

├─ ALL: Create assigned slides (Nov 28-30)

├─ ALL: Combine slides (Nov 30\)

└─ ALL: Practice presentation (Dec 1\)

    

🎯 PRESENTATION DUE: DEC 1

PHASE 3: INTEGRATION (Dec 2-8)

├─ Matthew: Finalize feature pipeline (Dec 3\)

├─ Lawson: Integrate full system (Dec 5\)

├─ Lawson: Run experiments (Dec 5-8)

└─ Matthew: Build baseline models

PHASE 4: PAPER WRITING (Dec 5-12)

├─ Victoria: Intro \+ lit review sections (Dec 5-10)

├─ Xiaoyu: Segmentation section (Dec 5-10)

├─ Matthew: Features section (Dec 8-10)

├─ Lawson: Experiments \+ results (Dec 8-10)

├─ ALL: Combine in LaTeX (Dec 11\)

└─ ALL: Final review \+ submission (Dec 12\)

🎯 PAPER \+ CODE DUE: DEC 12

---

## CRITICAL DEPENDENCIES

**The Big Three (Must Hit These Dates):**

1. **Xiaoyu's U-Net masks** (Nov 30\) → blocks everything downstream  
2. **Matthew's features** (Dec 3\) → blocks Lawson's integration  
3. **Lawson's integration** (Dec 5\) → blocks final experiments/results

---

## SHARED RESPONSIBILITIES

- Everyone reads Sequencer paper (by Nov 27\)  
- Presentation rehearsal together (Nov 30\)  
- Paper final review (Dec 11\)  
- Code testing \- everyone runs final pipeline (Dec 11\)

---

## WEEKLY MEETINGS?

**Week 1 (Nov 24-26):** Check dataset status, finalize feature design, lit review progress  
**Week 2 (Nov 28):** U-Net status, review presentation slides  
**Week 3 (Dec 3):** Integration check-in, paper assignments  
**Week 4 (Dec 10):** Final paper review, code testing

**Best Times:** Tuesday evenings, Sunday afternoons

---

## KEY LINKS

- [GitHub Repo](https://github.com/laws-smith-luna/MLIA_Final_Project)  
- [Google Slides](https://docs.google.com/presentation/d/1yCqg9VuO-2AwBvWxmIGuQKnzs1RT3iVz8qystFLWDdQ/edit?slide=id.p#slide=id.p)  
- LaTeX Platform

---

## QUICK RISK CHECKLIST

- [ ] Xiaoyu: U-Net training started by Nov 27 (buffer for issues)  
- [ ] Matthew: Have 2-3 feature extraction backup plans  
- [ ] Lawson: Study Sequencer early, ask for help if stuck  
- [ ] Victoria: Test LaTeX platform access by Dec 5  
- [ ] ALL: Keep Google Docs backup of paper text

