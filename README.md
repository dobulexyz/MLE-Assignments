# MLE-Assignments
A sample of teaching assignments that I have developed for the Machine Learning Engineering course at TU Berlin. 


## Summer 2026

During this semester I completely reworked the assignments for this course, adopting a repository-style format to better captures what is seen in the MLE discipline, and significantly increasing the amount of code students were required to write. This was done in part because previous assignments did not provide enough coding practice to students, as shown by lower exam scores and the consistent need to directly inspect a student's solution, as their solution could not run, rather than relying on automated tests. It should be noted that compared to `homework4-neural-networks` the other assignments were far less rigorous or lengthy. These other assignments are not included here, as I did not write them.

Additionally, unlike previous semesters, coding agents were **HEAVILY** used during the creation of these assignments. Nearly all output was written by Claude. Still, the format and planning of the assignments, their content, their inspection and refinement, as well as a significant amount of prompting were done by myself. I certainly think that the use of a coding agent saved me time and allowed me to set my ambitions much higher, but still, it at most reduced the time I had to put in by around 50%, overall.

Each of these assignments has a `task_prompts` folder, with a series of markdown files that the student is supposed to go through in order. Each prompt contains an introduction to the concept and component that the student will be implementing, or an introduction to a concept that is shown through an experiment. In most cases the code provided to the student is very well structure already, leaving the student to fill in function bodies. Earlier assignments had far more structure, while later ones tried to ask more of the student's own ability. Experiments are hard-coded into the assignments, so the student never has to adjust parameters in order to elicit a particular phenomenon, saving them time, and additionally to assure that each experiment can reasonably run on older hardware. The general requirement I used was that any experiment should not take more than 10 minutes to run one experiment on a common consumer-grade laptop that was released five years prior.

### `A4-neural-networks-in-numpy`

This assignment takes the student through implementing and training an MLP that is a written in `numpy`. In many ways it is a successor to `homework4-neural-networks`. The first task prompt for this assignment is literally the same introduction found in `homework4-neural-networks`. Additionally it introduces more advanced concepts, such using gradient norms or update magnitudes as a form of training diagnostic, regression tasks, data normalization and standardization, the Adam optimizer, dropout regularization, and first layer weight visualization (just for fun).


## Previous semesters

### `homework4-neural-networks`

This was written during the summer semester of 2024 and is the only assignment in this repository to be completely written by hand, or rather without the help of any coding agent. Within a jupyter notebook, it guides the student through the basic theory of the construction and training of an MLP, first using `numpy`, and later moving onto `torch`. This assignment as since been depreciated in the course.
