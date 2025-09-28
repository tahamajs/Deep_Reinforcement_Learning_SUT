# Deep Reinforcement Learning - Session 2 Exercise
#
# Markov Decision Processes and Value Functions**objective**: This Comprehensive Exercise Covers Fundamental Concepts of Reinforcement Learning Including Markov Decision Processes (mdps), Value Functions, Bellman Equations, and Policy Evaluation Methods.
#
#
# Topics COVERED:1. Introduction to Reinforcement Learning FRAMEWORK2. Markov Decision Processes (MDPS)3. Value Functions (state-value and ACTION-VALUE)4. Bellman EQUATIONS5. Policy Evaluation and IMPROVEMENT6. Practical Implementation with Gridworld Environment
#
#
# Learning Outcomes:by the End of This Exercise, You Will Understand:- the Mathematical Foundation of Mdps- How to Compute Value Functions- the Relationship between Policies and Value Functions- Implementation of Basic Rl Algorithms

# Table of Contents

- [Deep Reinforcement Learning - Session 2 Exercise## Markov Decision Processes and Value Functions**objective**: This Comprehensive Exercise Covers Fundamental Concepts of Reinforcement Learning Including Markov Decision Processes (mdps), Value Functions, Bellman Equations, and Policy Evaluation Methods.### Topics COVERED:1. Introduction to Reinforcement Learning FRAMEWORK2. Markov Decision Processes (MDPS)3. Value Functions (state-value and ACTION-VALUE)4. Bellman EQUATIONS5. Policy Evaluation and IMPROVEMENT6. Practical Implementation with Gridworld Environment### Learning Outcomes:by the End of This Exercise, You Will Understand:- the Mathematical Foundation of Mdps- How to Compute Value Functions- the Relationship between Policies and Value Functions- Implementation of Basic Rl Algorithms](#deep-reinforcement-learning---session-2-exercise-markov-decision-processes-and-value-functionsobjective-this-comprehensive-exercise-covers-fundamental-concepts-of-reinforcement-learning-including-markov-decision-processes-mdps-value-functions-bellman-equations-and-policy-evaluation-methods-topics-covered1-introduction-to-reinforcement-learning-framework2-markov-decision-processes-mdps3-value-functions-state-value-and-action-value4-bellman-equations5-policy-evaluation-and-improvement6-practical-implementation-with-gridworld-environment-learning-outcomesby-the-end-of-this-exercise-you-will-understand--the-mathematical-foundation-of-mdps--how-to-compute-value-functions--the-relationship-between-policies-and-value-functions--implementation-of-basic-rl-algorithms)
- [Table of Contents- [Deep Reinforcement Learning - Session 2 Exercise## Markov Decision Processes and Value Functions**objective**: This Comprehensive Exercise Covers Fundamental Concepts of Reinforcement Learning Including Markov Decision Processes (mdps), Value Functions, Bellman Equations, and Policy Evaluation Methods.### Topics COVERED:1. Introduction to Reinforcement Learning FRAMEWORK2. Markov Decision Processes (MDPS)3. Value Functions (state-value and ACTION-VALUE)4. Bellman EQUATIONS5. Policy Evaluation and IMPROVEMENT6. Practical Implementation with Gridworld Environment### Learning Outcomes:by the End of This Exercise, You Will Understand:- the Mathematical Foundation of Mdps- How to Compute Value Functions- the Relationship between Policies and Value Functions- Implementation of Basic Rl Algorithms](#deep-reinforcement-learning---session-2-exercise-markov-decision-processes-and-value-functionsobjective-this-comprehensive-exercise-covers-fundamental-concepts-of-reinforcement-learning-including-markov-decision-processes-mdps-value-functions-bellman-equations-and-policy-evaluation-methods-topics-covered1-introduction-to-reinforcement-learning-framework2-markov-decision-processes-mdps3-value-functions-state-value-and-action-value4-bellman-equations5-policy-evaluation-and-improvement6-practical-implementation-with-gridworld-environment-learning-outcomesby-the-end-of-this-exercise-you-will-understand--the-mathematical-foundation-of-mdps--how-to-compute-value-functions--the-relationship-between-policies-and-value-functions--implementation-of-basic-rl-algorithms)- [Part 1: Theoretical Foundation### 1.1 Reinforcement Learning Framework**definition:**reinforcement Learning Is a Computational Approach to Learning from Interaction. the Key Elements Are:- **agent**: the Learner and Decision Maker - the Entity That Makes Choices- **environment**: the World the Agent Interacts with - Everything outside the Agent- **state (s)**: Current Situation of the Agent - Describes the Current Circumstances- **action (a)**: Choices Available to the Agent - Decisions That Can Be Made- **reward (r)**: Numerical Feedback from Environment - Immediate Feedback Signal- **policy (π)**: Agent's Strategy for Choosing Actions - Mapping from States to Actions**real-world Analogy:**think of Rl like Learning to Drive:- **agent** = the Driver (you)- **environment** = Roads, Traffic, Weather Conditions- **state** = Current Speed, Position, Traffic around You- **actions** = Accelerate, Brake, Turn Left/right- **reward** = Positive for Safe Driving, Negative for Accidents- **policy** = Your Driving Strategy (cautious, Aggressive, Etc.)---### 1.2 Markov Decision Process (mdp)**definition:**an Mdp Is Defined by the Tuple (S, A, P, R, Γ) Where:- **s**: Set of States - All Possible Situations the Agent Can Encounter- **a**: Set of Actions - All Possible Decisions Available to the Agent- **p**: Transition Probability Function P(s'|s,a) - Probability of Moving to State S' Given Current State S and Action A- **r**: Reward Function R(s,a,s') - Immediate Reward Received for Transitioning from S to S' Via Action A- **γ**: Discount Factor (0 ≤ Γ ≤ 1) - Determines Importance of Future Rewards**markov Property:**the Future Depends Only on the Current State, Not on the History of How We Got There. MATHEMATICALLY:P(S*{T+1} = S' | S*t = S, A*t = A, S*{T-1}, A*{T-1}, ..., S*0, A*0) = P(S*{T+1} = S' | S*t = S, A*t = A)**intuition:**the Current State Contains All Information Needed to Make Optimal Decisions. the past Is Already "encoded" in the Current State.---### 1.3 Value Functions**state-value Function:**$$v^π(s) = \mathbb{e}*π[g*t | S_t = S]$$**interpretation:** Expected Total Reward When Starting from State S and Following Policy Π. It Answers: "HOW Good Is It to Be in This State?"**action-value Function:**$$q^π(s,a) = \mathbb{e}*π[g*t | S*t = S, A*t = A]$$**interpretation:** Expected Total Reward When Taking Action a in State S and Then Following Policy Π. It Answers: "HOW Good Is It to Take This Specific Action in This State?"**return (cumulative Reward):**$$g*t = R*{T+1} + ΓR*{T+2} + Γ^2R*{T+3} + ... = \SUM*{K=0}^{\INFTY} Γ^k R*{T+K+1}$$**WHY Discount Factor Γ?**- **Γ = 0**: Only Immediate Rewards Matter (myopic)- **Γ = 1**: All Future Rewards Are Equally Important- **0 < Γ < 1**: Future Rewards Are Discounted (realistic for Most Scenarios)---### 1.4 Bellman Equations**bellman Equation for State-value Function:**$$v^π(s) = \sum*a Π(a|s) \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**interpretation:** the Value of a State Equals the Immediate Reward Plus the Discounted Value of the Next State, Averaged over All Possible Actions and Transitions.**bellman Equation for Action-value Function:**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γ \sum*{a'} Π(a'|s')q^π(s',a')]$$**key Insight:** the Bellman Equations Express a Recursive Relationship - the Value of a State (OR State-action Pair) Depends on the Immediate Reward Plus the Discounted Value of Future States. This Is the Mathematical Foundation for Most Rl Algorithms.](#part-1-theoretical-foundation-11-reinforcement-learning-frameworkdefinitionreinforcement-learning-is-a-computational-approach-to-learning-from-interaction-the-key-elements-are--agent-the-learner-and-decision-maker---the-entity-that-makes-choices--environment-the-world-the-agent-interacts-with---everything-outside-the-agent--state-s-current-situation-of-the-agent---describes-the-current-circumstances--action-a-choices-available-to-the-agent---decisions-that-can-be-made--reward-r-numerical-feedback-from-environment---immediate-feedback-signal--policy-π-agents-strategy-for-choosing-actions---mapping-from-states-to-actionsreal-world-analogythink-of-rl-like-learning-to-drive--agent--the-driver-you--environment--roads-traffic-weather-conditions--state--current-speed-position-traffic-around-you--actions--accelerate-brake-turn-leftright--reward--positive-for-safe-driving-negative-for-accidents--policy--your-driving-strategy-cautious-aggressive-etc----12-markov-decision-process-mdpdefinitionan-mdp-is-defined-by-the-tuple-s-a-p-r-γ-where--s-set-of-states---all-possible-situations-the-agent-can-encounter--a-set-of-actions---all-possible-decisions-available-to-the-agent--p-transition-probability-function-pssa---probability-of-moving-to-state-s-given-current-state-s-and-action-a--r-reward-function-rsas---immediate-reward-received-for-transitioning-from-s-to-s-via-action-a--γ-discount-factor-0--γ--1---determines-importance-of-future-rewardsmarkov-propertythe-future-depends-only-on-the-current-state-not-on-the-history-of-how-we-got-there-mathematicallypst1--s--st--s-at--a-st-1-at-1--s0-a0--pst1--s--st--s-at--aintuitionthe-current-state-contains-all-information-needed-to-make-optimal-decisions-the-past-is-already-encoded-in-the-current-state----13-value-functionsstate-value-functionvπs--mathbbeπgt--s_t--sinterpretation-expected-total-reward-when-starting-from-state-s-and-following-policy-π-it-answers-how-good-is-it-to-be-in-this-stateaction-value-functionqπsa--mathbbeπgt--st--s-at--ainterpretation-expected-total-reward-when-taking-action-a-in-state-s-and-then-following-policy-π-it-answers-how-good-is-it-to-take-this-specific-action-in-this-statereturn-cumulative-rewardgt--rt1--γrt2--γ2rt3----sumk0infty-γk-rtk1why-discount-factor-γ--γ--0-only-immediate-rewards-matter-myopic--γ--1-all-future-rewards-are-equally-important--0--γ--1-future-rewards-are-discounted-realistic-for-most-scenarios----14-bellman-equationsbellman-equation-for-state-value-functionvπs--suma-πas-sums-pssarsas--γvπsinterpretation-the-value-of-a-state-equals-the-immediate-reward-plus-the-discounted-value-of-the-next-state-averaged-over-all-possible-actions-and-transitionsbellman-equation-for-action-value-functionqπsa--sums-pssarsas--γ-suma-πasqπsakey-insight-the-bellman-equations-express-a-recursive-relationship---the-value-of-a-state-or-state-action-pair-depends-on-the-immediate-reward-plus-the-discounted-value-of-future-states-this-is-the-mathematical-foundation-for-most-rl-algorithms)- [📚 Common Misconceptions and Clarifications**misconception 1: "THE Agent Knows the Environment Model"**reality:** in Most Rl Problems, the Agent Doesn't Know P(s'|s,a) or R(s,a,s'). This Is Called "model-free" Rl, Where the Agent Learns through Trial and Error.**misconception 2: "higher Rewards Are Always Better"**reality:** the Goal Is to Maximize *cumulative* Reward, Not Immediate Reward. Sometimes Taking a Small Immediate Reward Prevents Getting a Much Larger Future Reward.**misconception 3: "THE Policy Should Always Be Deterministic"****reality:** Stochastic Policies (that Output Probabilities) Are Often Better Because They Allow for Exploration and Can Be Optimal in Certain Environments.---### 🧠 Building Intuition: Restaurant Example**scenario:** You're Choosing Restaurants to Visit in a New City.**mdp Components:**- **states**: Your Hunger Level, Location, Time of Day, Budget- **actions**: Choose Restaurant A, B, C, or Cook at Home- **rewards**: Satisfaction from Food (immediate) + Health Effects (long-term)- **transitions**: How Your State Changes after Eating**value Functions:**- **v(hungry, Downtown, Evening)**: How Good Is This Situation Overall?- **q(hungry, Downtown, Evening, "restaurant A")**: How Good Is Choosing Restaurant a in This Situation?**policy Learning:** Initially Random Choices → Gradually Prefer Restaurants That Gave Good Experiences → Eventually Develop a Strategy That Considers Health, Taste, Cost, and Convenience.---### 🔧 Mathematical Properties and Theorems**theorem 1: Existence and Uniqueness of Value Functions**for Any Policy Π and Finite Mdp, There Exists a Unique Solution to the Bellman Equations.**theorem 2: Bellman Optimality Principle**a Policy Π Is Optimal If and Only If:$$v^π(s) = \max_a Q^π(s,a) \text{ for All } S \IN S$$**theorem 3: Policy Improvement Theorem**if Π' Is Greedy with Respect to V^π, Then V^π'(s) ≥ V^π(s) for All States S.**practical Implications:**- We Can Always Improve a Policy by Being Greedy with Respect to Its Value Function- There Always Exists an Optimal Policy (MAY Not Be Unique)- the Optimal Value Function Satisfies the Bellman Optimality Equations- **rewards**: +1 for Safe Driving, -10 for Accidents, -1 for Speeding Tickets- **policy**: Your Driving Strategy (aggressive, Conservative, Etc.)#### **why Markov Property Matters**the **markov Property** Means "THE Future Depends Only on the Present, Not the Past."**example**: in Chess, to Decide Your Next Move, You Only Need to See the Current Board Position. You Don't Need to Know How the Pieces Got There - the Complete Game History Is Irrelevant for Making the Optimal Next Move.**non-markov Example**: Predicting Tomorrow's Weather Based Only on Today's Weather (YOU Need Historical Patterns).#### **understanding the Discount Factor (γ)**the Discount Factor Determines How Much You Care About Future Rewards:- **Γ = 0**: "I Only Care About Immediate Rewards" (very Myopic)- Example: Only Caring About This Month's Salary, Not Career Growth - **Γ = 0.9**: "future Rewards Are Worth 90% of Immediate Rewards"- Example: Investing Money - You Value Future Returns but Prefer Sooner - **Γ = 1**: "future Rewards Are as Valuable as Immediate Rewards"- Example: Climate Change Actions - Long-term Benefits Matter Equally**mathematical Impact**:- Return G*t = R*{T+1} + ΓR*{T+2} + Γ²R*{T+3} + ...- with Γ=0.9: G*t = R*{T+1} + 0.9×R*{T+2} + 0.81×R*{T+3} + ...- Future Rewards Get Progressively Less Important](#-common-misconceptions-and-clarificationsmisconception-1-the-agent-knows-the-environment-modelreality-in-most-rl-problems-the-agent-doesnt-know-pssa-or-rsas-this-is-called-model-free-rl-where-the-agent-learns-through-trial-and-errormisconception-2-higher-rewards-are-always-betterreality-the-goal-is-to-maximize-cumulative-reward-not-immediate-reward-sometimes-taking-a-small-immediate-reward-prevents-getting-a-much-larger-future-rewardmisconception-3-the-policy-should-always-be-deterministicreality-stochastic-policies-that-output-probabilities-are-often-better-because-they-allow-for-exploration-and-can-be-optimal-in-certain-environments-----building-intuition-restaurant-examplescenario-youre-choosing-restaurants-to-visit-in-a-new-citymdp-components--states-your-hunger-level-location-time-of-day-budget--actions-choose-restaurant-a-b-c-or-cook-at-home--rewards-satisfaction-from-food-immediate--health-effects-long-term--transitions-how-your-state-changes-after-eatingvalue-functions--vhungry-downtown-evening-how-good-is-this-situation-overall--qhungry-downtown-evening-restaurant-a-how-good-is-choosing-restaurant-a-in-this-situationpolicy-learning-initially-random-choices--gradually-prefer-restaurants-that-gave-good-experiences--eventually-develop-a-strategy-that-considers-health-taste-cost-and-convenience-----mathematical-properties-and-theoremstheorem-1-existence-and-uniqueness-of-value-functionsfor-any-policy-π-and-finite-mdp-there-exists-a-unique-solution-to-the-bellman-equationstheorem-2-bellman-optimality-principlea-policy-π-is-optimal-if-and-only-ifvπs--max_a-qπsa-text-for-all--s-in-stheorem-3-policy-improvement-theoremif-π-is-greedy-with-respect-to-vπ-then-vπs--vπs-for-all-states-spractical-implications--we-can-always-improve-a-policy-by-being-greedy-with-respect-to-its-value-function--there-always-exists-an-optimal-policy-may-not-be-unique--the-optimal-value-function-satisfies-the-bellman-optimality-equations--rewards-1-for-safe-driving--10-for-accidents--1-for-speeding-tickets--policy-your-driving-strategy-aggressive-conservative-etc-why-markov-property-mattersthe-markov-property-means-the-future-depends-only-on-the-present-not-the-pastexample-in-chess-to-decide-your-next-move-you-only-need-to-see-the-current-board-position-you-dont-need-to-know-how-the-pieces-got-there---the-complete-game-history-is-irrelevant-for-making-the-optimal-next-movenon-markov-example-predicting-tomorrows-weather-based-only-on-todays-weather-you-need-historical-patterns-understanding-the-discount-factor-γthe-discount-factor-determines-how-much-you-care-about-future-rewards--γ--0-i-only-care-about-immediate-rewards-very-myopic--example-only-caring-about-this-months-salary-not-career-growth---γ--09-future-rewards-are-worth-90-of-immediate-rewards--example-investing-money---you-value-future-returns-but-prefer-sooner---γ--1-future-rewards-are-as-valuable-as-immediate-rewards--example-climate-change-actions---long-term-benefits-matter-equallymathematical-impact--return-gt--rt1--γrt2--γ²rt3----with-γ09-gt--rt1--09rt2--081rt3----future-rewards-get-progressively-less-important)- [🎮 Understanding Our Gridworld Environmentbefore We Dive into the Code, Let's Understand What We're Building:#### **the Gridworld SETUP**```(0,0) → → → (0,3) ↓ X X ↓ ↓ X ◯ ↓ (3,0) → → → (3,3) 🎯```**legend:**- `S` at (0,0): Starting Position- `🎯` at (3,3): Goal (treasure!)- `X`: Obstacles (walls or Pits)- `◯`: Regular Empty Spaces- Arrows: Possible Movements#### **why This Environment Is Perfect for LEARNING**1. **small & Manageable**: 4×4 Grid = 16 States (easy to VISUALIZE)2. **clear Objective**: Get from Start to GOAL3. **interesting Obstacles**: Forces Strategic THINKING4. **deterministic**: Same Action Always Leads to Same Result (FOR Now)#### **reward Structure Explained**- **goal Reward (+10)**: Big Positive Reward for Reaching the Treasure- **step Penalty (-0.1)**: Small Negative Reward for Each Move (encourages Efficiency)- **obstacle Penalty (-5)**: Big Negative Reward for Hitting Obstacles (safety First!)**why These Specific Values?**- Goal Reward Is Much Larger Than Step Penalty → Encourages Reaching the Goal- Obstacle Penalty Is Significant → Discourages Dangerous Moves- Step Penalty Is Small → Prevents Infinite Wandering without Being Too Harsh#### **state Representation**each State Is a Tuple (row, Column):- (0,0) = Top-left Corner- (3,3) = Bottom-right Corner - States Are like Gps Coordinates for Our Agent](#-understanding-our-gridworld-environmentbefore-we-dive-into-the-code-lets-understand-what-were-building-the-gridworld-setup00----03--x-x---x---30----33-legend--s-at-00-starting-position---at-33-goal-treasure--x-obstacles-walls-or-pits---regular-empty-spaces--arrows-possible-movements-why-this-environment-is-perfect-for-learning1-small--manageable-44-grid--16-states-easy-to-visualize2-clear-objective-get-from-start-to-goal3-interesting-obstacles-forces-strategic-thinking4-deterministic-same-action-always-leads-to-same-result-for-now-reward-structure-explained--goal-reward-10-big-positive-reward-for-reaching-the-treasure--step-penalty--01-small-negative-reward-for-each-move-encourages-efficiency--obstacle-penalty--5-big-negative-reward-for-hitting-obstacles-safety-firstwhy-these-specific-values--goal-reward-is-much-larger-than-step-penalty--encourages-reaching-the-goal--obstacle-penalty-is-significant--discourages-dangerous-moves--step-penalty-is-small--prevents-infinite-wandering-without-being-too-harsh-state-representationeach-state-is-a-tuple-row-column--00--top-left-corner--33--bottom-right-corner---states-are-like-gps-coordinates-for-our-agent)- [Part 2: Policy Definition and Evaluation### Exercise 2.1: Define Different Policies**definition:**a Policy Π(a|s) Defines the Probability of Taking Action a in State S. It's the Agent's Strategy for Choosing Actions.**mathematical Representation:**$$\pi(a|s) = P(\text{action} = a | \text{state} = S)$$**types of Policies:**- **deterministic Policy**: Π(a|s) ∈ {0, 1} - Always Chooses the Same Action in a Given State- **stochastic Policy**: Π(a|s) ∈ [0, 1] - Chooses Actions Probabilistically**policies We'll IMPLEMENT:**1. **random Policy**: Equal Probability for All Valid ACTIONS2. **greedy Policy**: Always Move towards the Goal 3. **custom Policy**: Your Own Strategic Policy---### Exercise 2.2: Policy Evaluation**definition:**policy Evaluation Computes the Value Function V^π(s) for a Given Policy Π. It Answers: "HOW Good Is This Policy?"**iterative Policy Evaluation ALGORITHM:**1. **initialize**: V(s) = 0 for All States S2. **repeat until Convergence**:- for Each State S:- V*new(s) = Σ*a Π(a|s) Σ*{s'} P(s'|s,a)[r(s,a,s') + ΓV(S')]3. **return**: Converged Value Function V**convergence Condition:**max*s |v*new(s) - V*old(s)| < Θ (where Θ Is a Small Threshold, E.g., 1E-6)**INTUITION:**WE Start with All State Values at Zero and Iteratively Update Them Based on the Bellman Equation until They Stabilize. It's like Repeatedly Asking "IF I Follow This Policy, How Much Reward Will I Get?" until the Answer Stops Changing.](#part-2-policy-definition-and-evaluation-exercise-21-define-different-policiesdefinitiona-policy-πas-defines-the-probability-of-taking-action-a-in-state-s-its-the-agents-strategy-for-choosing-actionsmathematical-representationpias--ptextaction--a--textstate--stypes-of-policies--deterministic-policy-πas--0-1---always-chooses-the-same-action-in-a-given-state--stochastic-policy-πas--0-1---chooses-actions-probabilisticallypolicies-well-implement1-random-policy-equal-probability-for-all-valid-actions2-greedy-policy-always-move-towards-the-goal-3-custom-policy-your-own-strategic-policy----exercise-22-policy-evaluationdefinitionpolicy-evaluation-computes-the-value-function-vπs-for-a-given-policy-π-it-answers-how-good-is-this-policyiterative-policy-evaluation-algorithm1-initialize-vs--0-for-all-states-s2-repeat-until-convergence--for-each-state-s--vnews--σa-πas-σs-pssarsas--γvs3-return-converged-value-function-vconvergence-conditionmaxs-vnews---volds--θ-where-θ-is-a-small-threshold-eg-1e-6intuitionwe-start-with-all-state-values-at-zero-and-iteratively-update-them-based-on-the-bellman-equation-until-they-stabilize-its-like-repeatedly-asking-if-i-follow-this-policy-how-much-reward-will-i-get-until-the-answer-stops-changing)- [🧭 Policy Deep Dive: Understanding Different Strategies**what Is a Policy?**a Policy Is like a Gps Navigation System for Our Agent. It Tells the Agent What to Do in Every Possible Situation.**mathematical Definition:**π(a|s) = Probability of Taking Action a When in State S---### 📋 Types of Policies We'll IMPLEMENT**1. Random Policy** 🎲**strategy:** "when in Doubt, Flip a Coin"**mathematical Definition:** Π(a|s) = 1/|VALID_ACTIONS| for All Valid Actions**example:** at State (1,0), If We Can Go [UP, Down, Right], Each Has 33.33% Probability**advantages:**- Explores All Possibilities Equally- Simple to Implement- Guarantees Exploration**disadvantages:**- Not Very Efficient- like Wandering Randomly in a Maze- No Learning from EXPERIENCE---**2. Greedy Policy** 🎯**strategy:** "always Move Closer to the Goal"**mathematical Definition:** Π(a|s) = 1 If a Minimizes Distance to Goal, 0 Otherwise**example:** at State (1,0), If Goal Is at (3,3), Prefer "down" and "right"**advantages:**- Very Efficient When It Works- Direct Path to Goal- Fast Convergence**disadvantages:**- Can Get Stuck in Local Optima- Might Walk into Obstacles- No Exploration of Alternative PATHS---**3. Custom Policy** 🎨**strategy:** Your Creative Combination of Strategies**examples:**- **epsilon-greedy**: 90% Greedy, 10% Random- **safety-first**: Avoid Actions That Lead near Obstacles- **wall-follower**: Stay Close to Boundaries---### 🎮 Real-world Analogies**policy Vs Strategy in Games:**think of Different Video Game Playing Styles:- **aggressive Player**: Always Attacks (deterministic Policy)- **defensive Player**: Always Defends (deterministic Policy)- **adaptive Player**: 70% Attack, 30% Defend (stochastic Policy)**why Stochastic Policies?**sometimes Randomness Helps:- **exploration**: Discover New Paths You Wouldn't Normally Try- **unpredictability**: in Competitive Games, Being Predictable Is Bad- **robustness**: Handle Uncertainty in the Environment**restaurant Choice Analogy:**- **random Policy**: Pick Restaurants Randomly- **greedy Policy**: Always Go to Your Current Favorite- **epsilon-greedy Policy**: Usually Go to Favorite, Sometimes Try Something New](#-policy-deep-dive-understanding-different-strategieswhat-is-a-policya-policy-is-like-a-gps-navigation-system-for-our-agent-it-tells-the-agent-what-to-do-in-every-possible-situationmathematical-definitionπas--probability-of-taking-action-a-when-in-state-s-----types-of-policies-well-implement1-random-policy-strategy-when-in-doubt-flip-a-coinmathematical-definition-πas--1valid_actions-for-all-valid-actionsexample-at-state-10-if-we-can-go-up-down-right-each-has-3333-probabilityadvantages--explores-all-possibilities-equally--simple-to-implement--guarantees-explorationdisadvantages--not-very-efficient--like-wandering-randomly-in-a-maze--no-learning-from-experience---2-greedy-policy-strategy-always-move-closer-to-the-goalmathematical-definition-πas--1-if-a-minimizes-distance-to-goal-0-otherwiseexample-at-state-10-if-goal-is-at-33-prefer-down-and-rightadvantages--very-efficient-when-it-works--direct-path-to-goal--fast-convergencedisadvantages--can-get-stuck-in-local-optima--might-walk-into-obstacles--no-exploration-of-alternative-paths---3-custom-policy-strategy-your-creative-combination-of-strategiesexamples--epsilon-greedy-90-greedy-10-random--safety-first-avoid-actions-that-lead-near-obstacles--wall-follower-stay-close-to-boundaries-----real-world-analogiespolicy-vs-strategy-in-gamesthink-of-different-video-game-playing-styles--aggressive-player-always-attacks-deterministic-policy--defensive-player-always-defends-deterministic-policy--adaptive-player-70-attack-30-defend-stochastic-policywhy-stochastic-policiessometimes-randomness-helps--exploration-discover-new-paths-you-wouldnt-normally-try--unpredictability-in-competitive-games-being-predictable-is-bad--robustness-handle-uncertainty-in-the-environmentrestaurant-choice-analogy--random-policy-pick-restaurants-randomly--greedy-policy-always-go-to-your-current-favorite--epsilon-greedy-policy-usually-go-to-favorite-sometimes-try-something-new)- [🔍 Understanding Policy Evaluation Step-by-steppolicy Evaluation Answers the Question: **"how Good Is Each State If I Follow This Policy?"**#### **the Intuition**imagine You're Evaluating Different Starting Positions in a Board Game:- Some Positions Are Naturally Better (closer to Winning)- Some Positions Are Worse (closer to Losing) - the "value" of a Position Depends on How Well You'll Do from There#### **mathematical Breakdown****the Bellman Equation for State Values:**```v^π(s) = Σ*a Π(a|s) × Σ*{s'} P(s'|s,a) × [r(s,a,s') + Γ × V^π(s')]```**let's Decode This Step by STEP:**1. **for Each Possible Action A**: Π(a|s) = "HOW Likely Am I to Take Action a in State S?"2. **for Each Possible Next State S'**: P(s'|s,a) = "IF I Take Action A, What's the Chance I End Up in State S'?"3. **calculate Immediate Reward + Future Value**: R(s,a,s') + Γ × V^π(s')- R(s,a,s') = "what Reward Do I Get Immediately?"- Γ × V^π(s') = "what's the Discounted Future VALUE?"4. **sum Everything Up**: This Gives the Expected Value of Being in State S#### **simple Example**let's Say We're at State (2,2) with a Random Policy:```python# Random Policy: Equal Probability for All Valid Actionsπ(up|s) = 0.25, Π(down|s) = 0.25, Π(left|s) = 0.25, Π(right|s) = 0.25# FOR Action "UP" → Next State (1,2)CONTRIBUTION*UP = 0.25 × 1.0 × (-0.1 + 0.9 × V(1,2))# FOR Action "down" → Next State (3,2)CONTRIBUTION*DOWN = 0.25 × 1.0 × (-0.1 + 0.9 × V(3,2))# ... and So on for Left and RIGHTV(2,2) = Contribution*up + Contribution*down + Contribution*left + Contribution*right```#### **why Iterative?**- We Start with V(s) = 0 for All States (initial Guess)- Each Iteration Improves Our Estimate Using Current Values- Eventually, Values Converge to True Values- like Asking "IF I Knew the Value of My Neighbors, What Would My Value Be?"#### **convergence Intuition**think of It like Gossip Spreading in a Neighborhood:- Initially, Nobody Knows the True "gossip" (values)- Each Iteration, Neighbors Share Information - Eventually, Everyone Converges to the Same True Story](#-understanding-policy-evaluation-step-by-steppolicy-evaluation-answers-the-question-how-good-is-each-state-if-i-follow-this-policy-the-intuitionimagine-youre-evaluating-different-starting-positions-in-a-board-game--some-positions-are-naturally-better-closer-to-winning--some-positions-are-worse-closer-to-losing---the-value-of-a-position-depends-on-how-well-youll-do-from-there-mathematical-breakdownthe-bellman-equation-for-state-valuesvπs--σa-πas--σs-pssa--rsas--γ--vπslets-decode-this-step-by-step1-for-each-possible-action-a-πas--how-likely-am-i-to-take-action-a-in-state-s2-for-each-possible-next-state-s-pssa--if-i-take-action-a-whats-the-chance-i-end-up-in-state-s3-calculate-immediate-reward--future-value-rsas--γ--vπs--rsas--what-reward-do-i-get-immediately--γ--vπs--whats-the-discounted-future-value4-sum-everything-up-this-gives-the-expected-value-of-being-in-state-s-simple-examplelets-say-were-at-state-22-with-a-random-policypython-random-policy-equal-probability-for-all-valid-actionsπups--025-πdowns--025-πlefts--025-πrights--025-for-action-up--next-state-12contributionup--025--10---01--09--v12-for-action-down--next-state-32contributiondown--025--10---01--09--v32--and-so-on-for-left-and-rightv22--contributionup--contributiondown--contributionleft--contributionright-why-iterative--we-start-with-vs--0-for-all-states-initial-guess--each-iteration-improves-our-estimate-using-current-values--eventually-values-converge-to-true-values--like-asking-if-i-knew-the-value-of-my-neighbors-what-would-my-value-be-convergence-intuitionthink-of-it-like-gossip-spreading-in-a-neighborhood--initially-nobody-knows-the-true-gossip-values--each-iteration-neighbors-share-information---eventually-everyone-converges-to-the-same-true-story)- [Exercise 2.3: Create Your Custom Policy**task**: Design and Implement Your Own Policy. Consider Strategies Like:- **wall-following**: Try to Stay Close to Walls- **risk-averse**: Avoid Obstacles with Higher Probability- **exploration-focused**: Balance between Moving towards Goal and Exploring**your Implementation Below**:](#exercise-23-create-your-custom-policytask-design-and-implement-your-own-policy-consider-strategies-like--wall-following-try-to-stay-close-to-walls--risk-averse-avoid-obstacles-with-higher-probability--exploration-focused-balance-between-moving-towards-goal-and-exploringyour-implementation-below)- [Part 3: Action-value Functions (q-functions)### Exercise 3.1: Computing Q-values**definition:**the Action-value Function Q^π(s,a) Represents the Expected Return When Taking Action a in State S and Then Following Policy Π.**key Question Q-functions Answer:**q-functions Answer: "what If I Take This Specific Action Here, Then Follow My Policy?"**mathematical Relationships:****v from Q (policy-weighted Average):**$$v^π(s) = \sum*a Π(a|s) Q^π(s,a)$$**q from V (bellman Backup):**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**bellman Equation for Q:**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γ \sum*{a'} Π(a'|s')q^π(s',a')]$$**intuition:**- **v(s)**: "HOW Good Is This State?" (following Current Policy)- **q(s,a)**: "HOW Good Is This Specific Action?" (then Following Policy)the V-q Relationship Is like Asking:- V: "HOW Well Will I Do from This Chess Position?"- Q: "HOW Well Will I Do If I Move My Queen Here, Then Play Normally?"](#part-3-action-value-functions-q-functions-exercise-31-computing-q-valuesdefinitionthe-action-value-function-qπsa-represents-the-expected-return-when-taking-action-a-in-state-s-and-then-following-policy-πkey-question-q-functions-answerq-functions-answer-what-if-i-take-this-specific-action-here-then-follow-my-policymathematical-relationshipsv-from-q-policy-weighted-averagevπs--suma-πas-qπsaq-from-v-bellman-backupqπsa--sums-pssarsas--γvπsbellman-equation-for-qqπsa--sums-pssarsas--γ-suma-πasqπsaintuition--vs-how-good-is-this-state-following-current-policy--qsa-how-good-is-this-specific-action-then-following-policythe-v-q-relationship-is-like-asking--v-how-well-will-i-do-from-this-chess-position--q-how-well-will-i-do-if-i-move-my-queen-here-then-play-normally)- [🎯 Q-functions Deep Dive: the "what If" Values**core Concept:**q-functions Provide Action-specific Evaluations, Allowing Us to Compare Different Choices Directly.---### 🍕 Restaurant Decision Analogy**scenario:** You're Choosing a Restaurant from Downtown Location.**value Functions:**- **v(downtown)** = 7.5 → "average Satisfaction from This Location with My Usual Choices"- **q(downtown, Pizza*place)** = 8.2 → "satisfaction If I Specifically Choose Pizza"- **q(downtown, Sushi*place)** = 6.8 → "satisfaction If I Specifically Choose Sushi"- **q(downtown, Burger*place)** = 7.1 → "satisfaction If I Specifically Choose Burgers"**policy Calculation:**if My Policy Is 50% Pizza, 30% Sushi, 20% Burgers:v(downtown) = 0.5×8.2 + 0.3×6.8 + 0.2×7.1 = 4.1 + 2.04 + 1.42 = 7.56 ✓---### 🧮 Mathematical Relationships EXPLAINED**1. V from Q (weighted Average):**$$v^π(s) = \sum*a Π(a|s) × Q^π(s,a)$$**interpretation:** State Value = Probability of Each Action × Value of That ACTION**2. Q from V (bellman Backup):**$$q^π(s,a) = \sum*{s'} P(s'|s,a) × [r(s,a,s') + Γv^π(s')]$$**interpretation:** Action Value = Immediate Reward + Discounted Future State Value---### 🔥 Why Q-functions MATTER**1. Direct Action Comparison:**- Q(s, Left) = 5.2 Vs Q(s, Right) = 7.8 → Choose Right!- No Need to Compute State Values FIRST**2. Policy Improvement:**- Π*new(s) = Argmax*a Q^π*old(s,a)- Directly Find the Best ACTION**3. Optimal Decision Making:**- Q*(s,a) Tells Us the Value of Each Action under Optimal Behavior- Essential for Q-learning Algorithms---### 📊 Visual Understandingthink of Q-values as Action-specific "heat Maps":- **hot Spots** (high Q-values): Good Actions to Take- **cold Spots** (LOW Q-values): Actions to Avoid- **separate Map for Each Action**: Q(s,↑), Q(s,↓), Q(s,←), Q(s,→)**gridworld Example:**- Q(state, "toward*goal") Typically Has Higher Values- Q(state, "toward*obstacle") Typically Has Lower Values- Q(state, "toward*wall") Often Has Negative Values - Like: "restaurant Satisfaction = Meal Quality + How I'll Feel Tomorrow"#### **why Q-functions MATTER**1. **better Decision Making**: Q-values Directly Tell Us Which Action Is Best- Max*a Q(s,a) Gives the Best Action in State S2. **policy Improvement**: We Can Improve Policies by Being Greedy W.r.t. Q-values- Π*new(s) = Argmax*a Q^Π_OLD(S,A)3. **action Comparison**: Compare Different Actions in the Same State- "should I Go Left or Right from Here?"#### **visual Understanding**think of Q-values as a "heat Map" for Each Action:- **hot Spots** (high Q-values): Good Actions to Take- **cold Spots** (LOW Q-values): Actions to Avoid - **different Maps for Each Action**: Q(s,up), Q(s,down), Q(s,left), Q(s,right)#### **common Confusion: V Vs Q**- **v(s)**: "HOW Good Is My Current Strategy from This Position?"- **q(s,a)**: "HOW Good Is This Specific Move, Then Using My Strategy?"it's like Asking:- V: "HOW Well Will I Do in This Chess Position?" - Q: "HOW Well Will I Do If I Move My Queen Here, Then Play Normally?"](#-q-functions-deep-dive-the-what-if-valuescore-conceptq-functions-provide-action-specific-evaluations-allowing-us-to-compare-different-choices-directly-----restaurant-decision-analogyscenario-youre-choosing-a-restaurant-from-downtown-locationvalue-functions--vdowntown--75--average-satisfaction-from-this-location-with-my-usual-choices--qdowntown-pizzaplace--82--satisfaction-if-i-specifically-choose-pizza--qdowntown-sushiplace--68--satisfaction-if-i-specifically-choose-sushi--qdowntown-burgerplace--71--satisfaction-if-i-specifically-choose-burgerspolicy-calculationif-my-policy-is-50-pizza-30-sushi-20-burgersvdowntown--0582--0368--0271--41--204--142--756------mathematical-relationships-explained1-v-from-q-weighted-averagevπs--suma-πas--qπsainterpretation-state-value--probability-of-each-action--value-of-that-action2-q-from-v-bellman-backupqπsa--sums-pssa--rsas--γvπsinterpretation-action-value--immediate-reward--discounted-future-state-value-----why-q-functions-matter1-direct-action-comparison--qs-left--52-vs-qs-right--78--choose-right--no-need-to-compute-state-values-first2-policy-improvement--πnews--argmaxa-qπoldsa--directly-find-the-best-action3-optimal-decision-making--qsa-tells-us-the-value-of-each-action-under-optimal-behavior--essential-for-q-learning-algorithms-----visual-understandingthink-of-q-values-as-action-specific-heat-maps--hot-spots-high-q-values-good-actions-to-take--cold-spots-low-q-values-actions-to-avoid--separate-map-for-each-action-qs-qs-qs-qsgridworld-example--qstate-towardgoal-typically-has-higher-values--qstate-towardobstacle-typically-has-lower-values--qstate-towardwall-often-has-negative-values---like-restaurant-satisfaction--meal-quality--how-ill-feel-tomorrow-why-q-functions-matter1-better-decision-making-q-values-directly-tell-us-which-action-is-best--maxa-qsa-gives-the-best-action-in-state-s2-policy-improvement-we-can-improve-policies-by-being-greedy-wrt-q-values--πnews--argmaxa-qπ_oldsa3-action-comparison-compare-different-actions-in-the-same-state--should-i-go-left-or-right-from-here-visual-understandingthink-of-q-values-as-a-heat-map-for-each-action--hot-spots-high-q-values-good-actions-to-take--cold-spots-low-q-values-actions-to-avoid---different-maps-for-each-action-qsup-qsdown-qsleft-qsright-common-confusion-v-vs-q--vs-how-good-is-my-current-strategy-from-this-position--qsa-how-good-is-this-specific-move-then-using-my-strategyits-like-asking--v-how-well-will-i-do-in-this-chess-position---q-how-well-will-i-do-if-i-move-my-queen-here-then-play-normally)- [Part 4: Policy Improvement and Policy Iteration### Exercise 4.1: Policy Improvement**definition:**given a Value Function V^π, We Can Improve the Policy by Being Greedy with Respect to the Action-value Function.**policy Improvement Formula:**$$π'(s) = \arg\max*a Q^π(s,a) = \arg\max*a \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**interpretation:** Choose the Action That Maximizes Expected Return from Each State.**policy Improvement Theorem:**if Π' Is Greedy with Respect to V^π, Then V^π'(s) ≥ V^π(s) for All States S.**translation:** "IF I Always Choose the Best Action Based on My Current Understanding, I Can Only Do Better (OR at Least as Well)."---### Exercise 4.2: Policy Iteration Algorithm**policy Iteration STEPS:**1. **initialize**: Start with Arbitrary Policy Π₀2. **repeat until Convergence**:- **policy Evaluation**: Compute V^π*k (solve Bellman Equation)- **policy Improvement**: Π*{K+1}(S) = Argmax*a Q^Π_K(S,A)3. **output**: Optimal Policy Π* and Value Function V***convergence Guarantee:** Policy Iteration Is Guaranteed to Converge to the Optimal Policy in Finite Time for Finite Mdps.**why It Works:**- Each Step Produces a Better (OR Equal) Policy- There Are Only Finitely Many Deterministic Policies- Must Eventually Reach Optimal Policy](#part-4-policy-improvement-and-policy-iteration-exercise-41-policy-improvementdefinitiongiven-a-value-function-vπ-we-can-improve-the-policy-by-being-greedy-with-respect-to-the-action-value-functionpolicy-improvement-formulaπs--argmaxa-qπsa--argmaxa-sums-pssarsas--γvπsinterpretation-choose-the-action-that-maximizes-expected-return-from-each-statepolicy-improvement-theoremif-π-is-greedy-with-respect-to-vπ-then-vπs--vπs-for-all-states-stranslation-if-i-always-choose-the-best-action-based-on-my-current-understanding-i-can-only-do-better-or-at-least-as-well----exercise-42-policy-iteration-algorithmpolicy-iteration-steps1-initialize-start-with-arbitrary-policy-π₀2-repeat-until-convergence--policy-evaluation-compute-vπk-solve-bellman-equation--policy-improvement-πk1s--argmaxa-qπ_ksa3-output-optimal-policy-π-and-value-function-vconvergence-guarantee-policy-iteration-is-guaranteed-to-converge-to-the-optimal-policy-in-finite-time-for-finite-mdpswhy-it-works--each-step-produces-a-better-or-equal-policy--there-are-only-finitely-many-deterministic-policies--must-eventually-reach-optimal-policy)- [🚀 Policy Improvement Deep Dive: Making Better Decisions**core Idea:** Use the Value Function to Make Better Action Choices.---### 📚 Learning Process Analogy**scenario:** You're Learning to Play Chess.**policy Evaluation:** "HOW Good Is My Current Playing Style?"- Analyze Your Current Strategy- Evaluate Typical Game Outcomes- Identify Strengths and Weaknesses**policy Improvement:** "HOW Can I Play Better?"- Look at Each Position Where You Made Suboptimal Moves- Replace Bad Moves with Better Alternatives- Update Your Playing Strategy**policy Iteration:** Repeat This Cycle until You Can't Improve Further.---### 🧮 Mathematical Foundations**policy Improvement Theorem:**if Π' Is Greedy W.r.t. V^π, Then V^π'(s) ≥ V^π(s) for All S.**proof INTUITION:**1. **greedy Action**: Choose a Such That Q^π(s,a) Is MAXIMIZED2. **definition**: Q^π(s,a) ≥ V^π(s) for the Chosen ACTION3. **new Policy**: Π'(s) Gives This Optimal ACTION4. **result**: V^π'(s) ≥ V^π(s)**why Greedy Improvement Works:**- Current Policy Chooses Actions with Average Value V^π(s)- Greedy Policy Chooses Action with Maximum Value Q^π(s,a)- Maximum ≥ Average, So New Policy Is Better---### 🔄 Policy Iteration: the Complete Cycle**step 1 - Policy Evaluation:** "HOW Good Is My Current Policy?"```v^π(s) ← Expected Return Following Π from State S```**step 2 - Policy Improvement:** "what's the Best Action in Each State?"```π'(s) ← Action That Maximizes Q^π(s,a)```**step 3 - Check Convergence:** "DID My Policy Change?"```if Π'(s) = Π(s) for All S: Stop (optimal Found)else: Π ← Π' and Repeat```---### 🎯 Key PROPERTIES**1. Monotonic IMPROVEMENT:**V^Π₀ ≤ V^Π₁ ≤ V^Π₂ ≤ ... ≤ V^Π**2. Finite Convergence:**algorithm Terminates in Finite Steps (FOR Finite MDPS)**3. Optimal Solution:**final Policy Π* Is Optimal: V^π* = V**4. Model-based:**requires Knowledge of Transition Probabilities P(s'|s,a) and Rewards R(s,a,s')think of a Student Improving Their Study STRATEGY:1. **current Strategy** (policy Π): "I Study Randomly for 2 HOURS"2. **evaluate Strategy** (policy Evaluation): "HOW Well Does This Work for Each Subject?" 3. **find Better Strategy** (policy Improvement): "math Needs 3 Hours, History Needs 1 HOUR"4. **repeat**: Keep Refining until No More Improvements Possible#### **mathematical Intuition**policy Improvement Theorem**: If Q^π(s,a) > V^π(s) for Some Action A, Then Taking Action a Is Better Than Following Policy Π.**translation**: "IF Doing Action a Gives Higher Value Than My Current Average, I Should Do Action a More Often!"**greedy Improvement**:```pythonπ*new(s) = Argmax*a Q^π(s,a)```"always Choose the Action with Highest Q-value"#### **why Does This Work?**monotonic Improvement**: Each Policy Improvement Step Makes the Policy at Least as Good, Usually Better.**proof Sketch**:- If We're Greedy W.r.t. Q^π, We Get V^π_new ≥ V^π- "IF I Always Choose the Best Available Action, I Can't Do Worse"#### **policy Iteration: the Complete Algorithm**the Cycle**:```random Policy → Evaluate → Improve → Evaluate → Improve → ... → Optimal Policy```**why It CONVERGES**:1. **finite State/action Space**: Limited Number of Possible POLICIES2. **monotonic Improvement**: Each Step Makes Policy Better (OR SAME)3. **NO Cycles**: Can't Go Backwards to a Worse POLICY4. **must Terminate**: Eventually Reach Optimal Policy#### **real-world Example: Learning to Drive**iteration 1**:- **policy**: "drive Slowly Everywhere" - **evaluation**: "safe but Inefficient on Highways"- **improvement**: "drive Fast on Highways, Slow in Neighborhoods"**iteration 2**:- **policy**: "speed Varies by Road Type"- **evaluation**: "good, but Inefficient in Traffic" - **improvement**: "also Consider Traffic Conditions"**final Policy**: "optimal Speed Based on Road Type, Traffic, Weather, Etc."#### **key INSIGHTS**1. **guaranteed Improvement**: Policy Iteration Always Finds the Optimal Policy (FOR Finite MDPS)2. **fast Convergence**: Usually Converges in Just a Few ITERATIONS3. **NO Exploration Needed**: Uses Complete Model Knowledge (unlike Q-learning LATER)4. **computational Cost**: Each Iteration Requires Solving the Bellman Equation#### **common Pitfalls**- **getting Stuck**: in Stochastic Environments, Might Need Exploration- **computational Cost**: Policy Evaluation Can Be Expensive - **model Required**: Need to Know P(s'|s,a) and R(s,a,s')](#-policy-improvement-deep-dive-making-better-decisionscore-idea-use-the-value-function-to-make-better-action-choices-----learning-process-analogyscenario-youre-learning-to-play-chesspolicy-evaluation-how-good-is-my-current-playing-style--analyze-your-current-strategy--evaluate-typical-game-outcomes--identify-strengths-and-weaknessespolicy-improvement-how-can-i-play-better--look-at-each-position-where-you-made-suboptimal-moves--replace-bad-moves-with-better-alternatives--update-your-playing-strategypolicy-iteration-repeat-this-cycle-until-you-cant-improve-further-----mathematical-foundationspolicy-improvement-theoremif-π-is-greedy-wrt-vπ-then-vπs--vπs-for-all-sproof-intuition1-greedy-action-choose-a-such-that-qπsa-is-maximized2-definition-qπsa--vπs-for-the-chosen-action3-new-policy-πs-gives-this-optimal-action4-result-vπs--vπswhy-greedy-improvement-works--current-policy-chooses-actions-with-average-value-vπs--greedy-policy-chooses-action-with-maximum-value-qπsa--maximum--average-so-new-policy-is-better-----policy-iteration-the-complete-cyclestep-1---policy-evaluation-how-good-is-my-current-policyvπs--expected-return-following-π-from-state-sstep-2---policy-improvement-whats-the-best-action-in-each-stateπs--action-that-maximizes-qπsastep-3---check-convergence-did-my-policy-changeif-πs--πs-for-all-s-stop-optimal-foundelse-π--π-and-repeat-----key-properties1-monotonic-improvementvπ₀--vπ₁--vπ₂----vπ2-finite-convergencealgorithm-terminates-in-finite-steps-for-finite-mdps3-optimal-solutionfinal-policy-π-is-optimal-vπ--v4-model-basedrequires-knowledge-of-transition-probabilities-pssa-and-rewards-rsasthink-of-a-student-improving-their-study-strategy1-current-strategy-policy-π-i-study-randomly-for-2-hours2-evaluate-strategy-policy-evaluation-how-well-does-this-work-for-each-subject-3-find-better-strategy-policy-improvement-math-needs-3-hours-history-needs-1-hour4-repeat-keep-refining-until-no-more-improvements-possible-mathematical-intuitionpolicy-improvement-theorem-if-qπsa--vπs-for-some-action-a-then-taking-action-a-is-better-than-following-policy-πtranslation-if-doing-action-a-gives-higher-value-than-my-current-average-i-should-do-action-a-more-oftengreedy-improvementpythonπnews--argmaxa-qπsaalways-choose-the-action-with-highest-q-value-why-does-this-workmonotonic-improvement-each-policy-improvement-step-makes-the-policy-at-least-as-good-usually-betterproof-sketch--if-were-greedy-wrt-qπ-we-get-vπ_new--vπ--if-i-always-choose-the-best-available-action-i-cant-do-worse-policy-iteration-the-complete-algorithmthe-cyclerandom-policy--evaluate--improve--evaluate--improve----optimal-policywhy-it-converges1-finite-stateaction-space-limited-number-of-possible-policies2-monotonic-improvement-each-step-makes-policy-better-or-same3-no-cycles-cant-go-backwards-to-a-worse-policy4-must-terminate-eventually-reach-optimal-policy-real-world-example-learning-to-driveiteration-1--policy-drive-slowly-everywhere---evaluation-safe-but-inefficient-on-highways--improvement-drive-fast-on-highways-slow-in-neighborhoodsiteration-2--policy-speed-varies-by-road-type--evaluation-good-but-inefficient-in-traffic---improvement-also-consider-traffic-conditionsfinal-policy-optimal-speed-based-on-road-type-traffic-weather-etc-key-insights1-guaranteed-improvement-policy-iteration-always-finds-the-optimal-policy-for-finite-mdps2-fast-convergence-usually-converges-in-just-a-few-iterations3-no-exploration-needed-uses-complete-model-knowledge-unlike-q-learning-later4-computational-cost-each-iteration-requires-solving-the-bellman-equation-common-pitfalls--getting-stuck-in-stochastic-environments-might-need-exploration--computational-cost-policy-evaluation-can-be-expensive---model-required-need-to-know-pssa-and-rsas)- [Part 5: Experiments and Analysis### Exercise 5.1: Effect of Discount Factor (γ)**definition:**the Discount Factor Γ Determines How Much We Value Future Rewards Compared to Immediate Rewards.**mathematical Impact:**$$g*t = R*{T+1} + ΓR*{T+2} + Γ^2R*{T+3} + ... = \SUM*{K=0}^{\INFTY} Γ^k R*{T+K+1}$$**INTERPRETATION of Different Values:**- **Γ = 0**: Only Immediate Rewards Matter (myopic Behavior)- **Γ = 1**: All Future Rewards Equally Important (infinite Horizon)- **0 < Γ < 1**: Future Rewards Are Discounted (realistic)**task:** Experiment with Different Discount Factors and Analyze Their Effect on the Optimal Policy.**research QUESTIONS:**1. How Does Γ Affect the Optimal POLICY?2. Which Γ Values Lead to Faster CONVERGENCE?3. What Happens to State Values as Γ Changes?](#part-5-experiments-and-analysis-exercise-51-effect-of-discount-factor-γdefinitionthe-discount-factor-γ-determines-how-much-we-value-future-rewards-compared-to-immediate-rewardsmathematical-impactgt--rt1--γrt2--γ2rt3----sumk0infty-γk-rtk1interpretation-of-different-values--γ--0-only-immediate-rewards-matter-myopic-behavior--γ--1-all-future-rewards-equally-important-infinite-horizon--0--γ--1-future-rewards-are-discounted-realistictask-experiment-with-different-discount-factors-and-analyze-their-effect-on-the-optimal-policyresearch-questions1-how-does-γ-affect-the-optimal-policy2-which-γ-values-lead-to-faster-convergence3-what-happens-to-state-values-as-γ-changes)- [💰 Discount Factor Deep Dive: Balancing Present Vs Future**core Concept:** the Discount Factor Γ Controls the Agent's "patience" or Time Preference.---### ⏰ Time Value of Rewards**financial Analogy:**just like Money, Rewards Have "time Value":- $100 Today Vs $100 in 10 Years → Most Prefer Today (inflation, Uncertainty)- +10 Reward Now Vs +10 Reward in 100 Time Steps → Usually Prefer Immediate**mathematical Effect:**- **Γ = 0.1**: Reward 10 Steps Away Is Worth 0.1¹⁰ = 0.0000000001 of Current Reward- **Γ = 0.9**: Reward 10 Steps Away Is Worth 0.9¹⁰ = 0.35 of Current Reward- **Γ = 0.99**: Reward 10 Steps Away Is Worth 0.99¹⁰ = 0.90 of Current Reward---### 🌎 Real-world Analogies**γ = 0.1 (very Impatient/myopic):**- 🍕 "I Want Pizza Now, Don't Care About Health Consequences"- 💳 "BUY with Credit Card, Ignore Interest Charges"- 🚗 "take Fastest Route, Ignore Traffic Fines"**γ = 0.5 (moderately Patient):**- 🏃 "exercise Sometimes for Health Benefits"- 💰 "save Some Money, Spend Some Now"- 📚 "study When Motivated, Party When Not"**γ = 0.9 (balanced):**- 💪 "exercise Regularly for Long-term Health"- 🎓 "study Hard Now for Career Benefits Later"- 💰 "invest Consistently for Retirement"**γ = 0.99 (very Patient):**- 🌱 "plant Trees for Future Generations"- 🏠 "BUY House as Long-term Investment"- 🌍 "address Climate Change for Distant Future"---### 📊 Effect on Optimal Policy**low Γ (myopic Behavior):**- Takes Shortest Immediate Path to Reward- Ignores Long-term Consequences- May Get Stuck in Local Optima- Fast Convergence but Potentially Poor Solutions**high Γ (farsighted Behavior):**- Considers Long-term Consequences- May Take Longer Paths for Better Future Outcomes- Explores More Thoroughly- Slower Convergence but Better Final Solutions**in Gridworld Context:**- **low Γ**: Rushes toward Goal, Ignoring Obstacles- **high Γ**: Carefully Plans Path, Avoids Risky Moves#### **mathematical Impact**return Formula**: G*t = R*{T+1} + ΓR*{T+2} + Γ²R*{T+3} + Γ³R*{T+4} + ...**examples**:**γ = 0.9** (patient Agent):- G*t = R*{T+1} + 0.9×R*{T+2} + 0.81×R*{T+3} + 0.729×R*{T+4} + ...- Reward in 1 Step: Worth 100% of Immediate Reward- Reward in 2 Steps: Worth 90% of Immediate Reward - Reward in 3 Steps: Worth 81% of Immediate Reward- Reward in 10 Steps: Worth 35% of Immediate Reward**γ = 0.1** (impatient Agent):- G*t = R*{T+1} + 0.1×R*{T+2} + 0.01×R*{T+3} + 0.001×R_{T+4} + ...- Reward in 2 Steps: Worth Only 10% of Immediate Reward- Reward in 3 Steps: Worth Only 1% of Immediate Reward- Very Myopic - Only Cares About Next Few Steps#### **real-world Analogies**γ = 0.1** (very Impatient):- 🍕 "I Want Pizza Now, Don't Care About Health Consequences"- 📱 "BUY the Cheapest Phone, Ignore Long-term Durability" - 🚗 "take the Fastest Route, Ignore Traffic Fines"**γ = 0.9** (balanced):- 💪 "exercise Now for Health Benefits Later"- 🎓 "study Hard Now for Career Benefits Later"- 💰 "invest Money for Retirement"**γ = 0.99** (very Patient):- 🌱 "plant Trees for Future Generations"- 🏠 "BUY a House as Long-term Investment"- 🌍 "address Climate Change for Distant Future"#### **effect on Optimal Policy**low Γ (myopic Behavior)**:- Takes Shortest Path to Goal- Ignores Long-term Consequences - Might Take Dangerous Shortcuts- Policy: "rush to Goal, Avoid Obstacles Minimally"**high Γ (farsighted Behavior)**:- Takes Safer, Longer Paths- Values Long-term Safety- More Conservative Decisions- Policy: "GET to Goal Safely, Even If It Takes Longer"#### **choosing Γ in PRACTICE**CONSIDER**:1. **problem Horizon**: Short-term Tasks → Lower Γ, Long-term Tasks → Higher Γ2. **uncertainty**: More Uncertain Future → Lower Γ3. **safety**: Safety-critical Applications → Higher Γ4. **computational**: Higher Γ → Slower Convergence**common Values**:- **Γ = 0.9**: General Purpose, Good Balance- **Γ = 0.95-0.99**: Long-term Planning Tasks- **Γ = 0.1-0.5**: Short-term Reactive Tasks- **Γ = 1.0**: Infinite Horizon, Theoretical Studies (CAN Cause Issues)#### **debugging with Γ**if Your Agent:- **ignores Long-term Rewards**: Increase Γ- **IS Too Conservative**: Decrease Γ - **won't Converge**: Check If Γ Is Too Close to 1- **makes Random Decisions**: Γ Might Be Too Low](#-discount-factor-deep-dive-balancing-present-vs-futurecore-concept-the-discount-factor-γ-controls-the-agents-patience-or-time-preference-----time-value-of-rewardsfinancial-analogyjust-like-money-rewards-have-time-value--100-today-vs-100-in-10-years--most-prefer-today-inflation-uncertainty--10-reward-now-vs-10-reward-in-100-time-steps--usually-prefer-immediatemathematical-effect--γ--01-reward-10-steps-away-is-worth-01¹⁰--00000000001-of-current-reward--γ--09-reward-10-steps-away-is-worth-09¹⁰--035-of-current-reward--γ--099-reward-10-steps-away-is-worth-099¹⁰--090-of-current-reward-----real-world-analogiesγ--01-very-impatientmyopic---i-want-pizza-now-dont-care-about-health-consequences---buy-with-credit-card-ignore-interest-charges---take-fastest-route-ignore-traffic-finesγ--05-moderately-patient---exercise-sometimes-for-health-benefits---save-some-money-spend-some-now---study-when-motivated-party-when-notγ--09-balanced---exercise-regularly-for-long-term-health---study-hard-now-for-career-benefits-later---invest-consistently-for-retirementγ--099-very-patient---plant-trees-for-future-generations---buy-house-as-long-term-investment---address-climate-change-for-distant-future-----effect-on-optimal-policylow-γ-myopic-behavior--takes-shortest-immediate-path-to-reward--ignores-long-term-consequences--may-get-stuck-in-local-optima--fast-convergence-but-potentially-poor-solutionshigh-γ-farsighted-behavior--considers-long-term-consequences--may-take-longer-paths-for-better-future-outcomes--explores-more-thoroughly--slower-convergence-but-better-final-solutionsin-gridworld-context--low-γ-rushes-toward-goal-ignoring-obstacles--high-γ-carefully-plans-path-avoids-risky-moves-mathematical-impactreturn-formula-gt--rt1--γrt2--γ²rt3--γ³rt4--examplesγ--09-patient-agent--gt--rt1--09rt2--081rt3--0729rt4----reward-in-1-step-worth-100-of-immediate-reward--reward-in-2-steps-worth-90-of-immediate-reward---reward-in-3-steps-worth-81-of-immediate-reward--reward-in-10-steps-worth-35-of-immediate-rewardγ--01-impatient-agent--gt--rt1--01rt2--001rt3--0001r_t4----reward-in-2-steps-worth-only-10-of-immediate-reward--reward-in-3-steps-worth-only-1-of-immediate-reward--very-myopic---only-cares-about-next-few-steps-real-world-analogiesγ--01-very-impatient---i-want-pizza-now-dont-care-about-health-consequences---buy-the-cheapest-phone-ignore-long-term-durability----take-the-fastest-route-ignore-traffic-finesγ--09-balanced---exercise-now-for-health-benefits-later---study-hard-now-for-career-benefits-later---invest-money-for-retirementγ--099-very-patient---plant-trees-for-future-generations---buy-a-house-as-long-term-investment---address-climate-change-for-distant-future-effect-on-optimal-policylow-γ-myopic-behavior--takes-shortest-path-to-goal--ignores-long-term-consequences---might-take-dangerous-shortcuts--policy-rush-to-goal-avoid-obstacles-minimallyhigh-γ-farsighted-behavior--takes-safer-longer-paths--values-long-term-safety--more-conservative-decisions--policy-get-to-goal-safely-even-if-it-takes-longer-choosing-γ-in-practiceconsider1-problem-horizon-short-term-tasks--lower-γ-long-term-tasks--higher-γ2-uncertainty-more-uncertain-future--lower-γ3-safety-safety-critical-applications--higher-γ4-computational-higher-γ--slower-convergencecommon-values--γ--09-general-purpose-good-balance--γ--095-099-long-term-planning-tasks--γ--01-05-short-term-reactive-tasks--γ--10-infinite-horizon-theoretical-studies-can-cause-issues-debugging-with-γif-your-agent--ignores-long-term-rewards-increase-γ--is-too-conservative-decrease-γ---wont-converge-check-if-γ-is-too-close-to-1--makes-random-decisions-γ-might-be-too-low)- [Exercise 5.2: Modified Environment Experiments**task A**: Modify the Reward Structure and Analyze How It Affects the Optimal Policy:- Change Step Reward from -0.1 to -1.0 (higher Cost for Each Step)- Change Goal Reward from 10 to 5- Add Positive Rewards for Certain States**task B**: Experiment with Different Obstacle Configurations:- Remove Some Obstacles- Add More Obstacles- Change Obstacle Positions**task C**: Test with Different Starting Positions and Analyze Convergence.](#exercise-52-modified-environment-experimentstask-a-modify-the-reward-structure-and-analyze-how-it-affects-the-optimal-policy--change-step-reward-from--01-to--10-higher-cost-for-each-step--change-goal-reward-from-10-to-5--add-positive-rewards-for-certain-statestask-b-experiment-with-different-obstacle-configurations--remove-some-obstacles--add-more-obstacles--change-obstacle-positionstask-c-test-with-different-starting-positions-and-analyze-convergence)- [Part 6: Summary and Key Takeaways### What We've LEARNED**1. Markov Decision Processes (mdps):**- **framework**: Sequential Decision Making under Uncertainty- **components**: (S, A, P, R, Γ) - States, Actions, Transitions, Rewards, Discount- **markov Property**: Future Depends Only on Current State, Not History- **foundation**: Mathematical Basis for All Rl ALGORITHMS**2. Value Functions:**- **v^π(s)**: Expected Return Starting from State S Following Policy Π - **q^π(s,a)**: Expected Return Taking Action a in State S, Then Following Π- **relationship**: V^π(s) = Σ*a Π(a|s) Q^π(s,a)- **purpose**: Measure "goodness" of States and ACTIONS**3. Bellman Equations:**- **for V**: V^π(s) = Σ*a Π(a|s) Σ*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]- **for Q**: Q^π(s,a) = Σ*{s'} P(s'|s,a)[r(s,a,s') + Γ Σ*{a'} Π(a'|s')q^π(s',a')]- **significance**: Recursive Relationship Enabling Dynamic Programming SOLUTIONS**4. Policy Evaluation:**- **algorithm**: Iterative Method to Compute V^π Given Policy Π- **convergence**: Guaranteed for Finite Mdps with Γ < 1- **application**: Foundation for Policy Iteration and Value ITERATION**5. Policy Improvement:**- **theorem**: Greedy Policy W.r.t. V^π Is at Least as Good as Π- **formula**: Π'(s) = Argmax*a Q^π(s,a)- **monotonicity**: Each Improvement Step Yields Better or Equal POLICY**6. Policy Iteration:**- **algorithm**: Alternates between Evaluation and Improvement- **guarantee**: Converges to Optimal Policy Π*- **efficiency**: Usually Converges in Few Iterations---### Key Insights from Experiments**discount Factor (Γ) Effects:**- **low Γ**: Myopic Behavior, Focuses on Immediate Rewards- **high Γ**: Farsighted Behavior, Considers Long-term Consequences- **trade-off**: Convergence Speed Vs Solution Quality**environment Structure Impact:**- **reward Structure**: Significantly Affects Optimal Policy- **obstacles**: Create Navigation Challenges Requiring Planning- **starting Position**: Can Influence Learning Dynamics**algorithm Characteristics:**- **model-based**: Requires Knowledge of P(s'|s,a) and R(s,a,s')- **exact Solution**: Finds Truly Optimal Policy (unlike Approximate Methods)- **computational Cost**: Scales with State Space Size---### Connections to Advanced Topics**what This Enables:**- **value Iteration**: Direct Optimization of Value Function- **q-learning**: Model-free Learning of Action-value Functions- **deep Rl**: Neural Network Function Approximation- **policy Gradients**: Direct Policy Optimization Methods**next Steps in Learning:**- **temporal Difference Learning**: Learn from Incomplete Episodes- **function Approximation**: Handle Large/continuous State Spaces- **exploration Vs Exploitation**: Balance Learning and Performance- **multi-agent Systems**: Multiple Learning Agents Interacting---### Reflection Questions**theoretical UNDERSTANDING:**1. How Would Stochastic Transitions Affect the Optimal POLICY?2. What Happens with Continuous State or Action SPACES?3. How Do We Handle Unknown Environment DYNAMICS?4. What Are Computational Limits for Large State Spaces?**practical APPLICATIONS:**1. How Could You Apply Mdps to Real-world Decision PROBLEMS?2. What Modifications Would Be Needed for Competitive SCENARIOS?3. How Would You Handle Partially Observable ENVIRONMENTS?4. What Safety Considerations Are Important in Rl Applications?](#part-6-summary-and-key-takeaways-what-weve-learned1-markov-decision-processes-mdps--framework-sequential-decision-making-under-uncertainty--components-s-a-p-r-γ---states-actions-transitions-rewards-discount--markov-property-future-depends-only-on-current-state-not-history--foundation-mathematical-basis-for-all-rl-algorithms2-value-functions--vπs-expected-return-starting-from-state-s-following-policy-π---qπsa-expected-return-taking-action-a-in-state-s-then-following-π--relationship-vπs--σa-πas-qπsa--purpose-measure-goodness-of-states-and-actions3-bellman-equations--for-v-vπs--σa-πas-σs-pssarsas--γvπs--for-q-qπsa--σs-pssarsas--γ-σa-πasqπsa--significance-recursive-relationship-enabling-dynamic-programming-solutions4-policy-evaluation--algorithm-iterative-method-to-compute-vπ-given-policy-π--convergence-guaranteed-for-finite-mdps-with-γ--1--application-foundation-for-policy-iteration-and-value-iteration5-policy-improvement--theorem-greedy-policy-wrt-vπ-is-at-least-as-good-as-π--formula-πs--argmaxa-qπsa--monotonicity-each-improvement-step-yields-better-or-equal-policy6-policy-iteration--algorithm-alternates-between-evaluation-and-improvement--guarantee-converges-to-optimal-policy-π--efficiency-usually-converges-in-few-iterations----key-insights-from-experimentsdiscount-factor-γ-effects--low-γ-myopic-behavior-focuses-on-immediate-rewards--high-γ-farsighted-behavior-considers-long-term-consequences--trade-off-convergence-speed-vs-solution-qualityenvironment-structure-impact--reward-structure-significantly-affects-optimal-policy--obstacles-create-navigation-challenges-requiring-planning--starting-position-can-influence-learning-dynamicsalgorithm-characteristics--model-based-requires-knowledge-of-pssa-and-rsas--exact-solution-finds-truly-optimal-policy-unlike-approximate-methods--computational-cost-scales-with-state-space-size----connections-to-advanced-topicswhat-this-enables--value-iteration-direct-optimization-of-value-function--q-learning-model-free-learning-of-action-value-functions--deep-rl-neural-network-function-approximation--policy-gradients-direct-policy-optimization-methodsnext-steps-in-learning--temporal-difference-learning-learn-from-incomplete-episodes--function-approximation-handle-largecontinuous-state-spaces--exploration-vs-exploitation-balance-learning-and-performance--multi-agent-systems-multiple-learning-agents-interacting----reflection-questionstheoretical-understanding1-how-would-stochastic-transitions-affect-the-optimal-policy2-what-happens-with-continuous-state-or-action-spaces3-how-do-we-handle-unknown-environment-dynamics4-what-are-computational-limits-for-large-state-spacespractical-applications1-how-could-you-apply-mdps-to-real-world-decision-problems2-what-modifications-would-be-needed-for-competitive-scenarios3-how-would-you-handle-partially-observable-environments4-what-safety-considerations-are-important-in-rl-applications)- [🧠 Common Misconceptions and Intuitive Understandingbefore We Wrap Up, Let's Address Some Common Confusions and Solidify Understanding:#### **❌ Common MISCONCEPTIONS**1. "value Functions Are Just Rewards"**- ❌ Wrong: V(s) ≠ R(s) - ✅ Correct: V(s) = Expected Total Future Reward from State S- 🔍 Think: V(s) Is like Your Bank Account Balance, R(s) Is Your Daily INCOME**2. "q(s,a) Tells Me the Best Action"**- ❌ Wrong: Q(s,a) Is Not Binary Good/bad- ✅ Correct: Q(s,a) Is the Expected Value of Taking Action A- 🔍 Think: Compare Q-values to Choose Best Action: Argmax_a Q(S,A)**3. "policy Iteration Always Takes Many Steps"**- ❌ Wrong: Often Converges in 2-4 Iterations- ✅ Correct: Convergence Is Usually Very Fast- 🔍 Think: Once You Find a Good Strategy, Small Improvements Are ENOUGH**4. "random Policy Is Always Bad"**- ❌ Wrong: Random Policy Can Be Good for Exploration- ✅ Correct: Depends on Environment and Goals- 🔍 Think: Sometimes Trying New Things Leads to Better Discoveries#### **🎯 Key Intuitions to REMEMBER**1. the Big Picture Flow**:```environment → Policy → Actions → Rewards → Better Policy → REPEAT```**2. Value Functions as Gps**:- V(s): "HOW Good Is This Location Overall?"- Q(s,a): "HOW Good Is Taking This Road from This LOCATION?"**3. Bellman Equations as Consistency**:- "MY Value Should Equal Immediate Reward + Discounted Future Value"- Like: "MY Wealth = Today's Income + Tomorrow's WEALTH"**4. Policy Improvement as Learning**:- "IF I Know What Each Action Leads To, I Can Choose Better Actions"- Like: "IF I Know Exam Results for Each Study Method, I Can Study Better"#### **🔧 Troubleshooting Guide****if Values Don't Converge**:- Check If Γ < 1 - Reduce Convergence Threshold (theta)- Check for Bugs in Transition Probabilities**if Policy Doesn't Improve**:- Environment Might Be Too Simple (already Optimal)- Check Reward Structure - Might Need More Differentiation- Verify Policy Improvement Logic**if Results Seem Weird**:- Visualize Value Functions and Policies- Start with Simpler Environment- Check Reward Signs (positive/negative)#### **🚀 Connecting to Future Topics**what We Learned Here Enables:- **value Iteration**: Direct Value Optimization (next Week!)- **q-learning**: Learn Q-values without Knowing the Model- **deep Rl**: Use Neural Networks to Handle Large State Spaces- **policy Gradients**: Directly Optimize the Policy Parameters#### **🎭 the Rl Mindset**think like an Rl AGENT:1. **observe** Your Current Situation (STATE)2. **consider** Your Options (actions) 3. **predict** Outcomes (USE Your MODEL/EXPERIENCE)4. **choose** the Best Option (POLICY)5. **learn** from Results (update VALUES/POLICY)6. **repeat** until Masterythis Mindset Applies To:- Career Decisions- Investment Choices - Game Strategies- Daily Life Optimization](#-common-misconceptions-and-intuitive-understandingbefore-we-wrap-up-lets-address-some-common-confusions-and-solidify-understanding--common-misconceptions1-value-functions-are-just-rewards---wrong-vs--rs----correct-vs--expected-total-future-reward-from-state-s---think-vs-is-like-your-bank-account-balance-rs-is-your-daily-income2-qsa-tells-me-the-best-action---wrong-qsa-is-not-binary-goodbad---correct-qsa-is-the-expected-value-of-taking-action-a---think-compare-q-values-to-choose-best-action-argmax_a-qsa3-policy-iteration-always-takes-many-steps---wrong-often-converges-in-2-4-iterations---correct-convergence-is-usually-very-fast---think-once-you-find-a-good-strategy-small-improvements-are-enough4-random-policy-is-always-bad---wrong-random-policy-can-be-good-for-exploration---correct-depends-on-environment-and-goals---think-sometimes-trying-new-things-leads-to-better-discoveries--key-intuitions-to-remember1-the-big-picture-flowenvironment--policy--actions--rewards--better-policy--repeat2-value-functions-as-gps--vs-how-good-is-this-location-overall--qsa-how-good-is-taking-this-road-from-this-location3-bellman-equations-as-consistency--my-value-should-equal-immediate-reward--discounted-future-value--like-my-wealth--todays-income--tomorrows-wealth4-policy-improvement-as-learning--if-i-know-what-each-action-leads-to-i-can-choose-better-actions--like-if-i-know-exam-results-for-each-study-method-i-can-study-better--troubleshooting-guideif-values-dont-converge--check-if-γ--1---reduce-convergence-threshold-theta--check-for-bugs-in-transition-probabilitiesif-policy-doesnt-improve--environment-might-be-too-simple-already-optimal--check-reward-structure---might-need-more-differentiation--verify-policy-improvement-logicif-results-seem-weird--visualize-value-functions-and-policies--start-with-simpler-environment--check-reward-signs-positivenegative--connecting-to-future-topicswhat-we-learned-here-enables--value-iteration-direct-value-optimization-next-week--q-learning-learn-q-values-without-knowing-the-model--deep-rl-use-neural-networks-to-handle-large-state-spaces--policy-gradients-directly-optimize-the-policy-parameters--the-rl-mindsetthink-like-an-rl-agent1-observe-your-current-situation-state2-consider-your-options-actions-3-predict-outcomes-use-your-modelexperience4-choose-the-best-option-policy5-learn-from-results-update-valuespolicy6-repeat-until-masterythis-mindset-applies-to--career-decisions--investment-choices---game-strategies--daily-life-optimization)](#table-of-contents--deep-reinforcement-learning---session-2-exercise-markov-decision-processes-and-value-functionsobjective-this-comprehensive-exercise-covers-fundamental-concepts-of-reinforcement-learning-including-markov-decision-processes-mdps-value-functions-bellman-equations-and-policy-evaluation-methods-topics-covered1-introduction-to-reinforcement-learning-framework2-markov-decision-processes-mdps3-value-functions-state-value-and-action-value4-bellman-equations5-policy-evaluation-and-improvement6-practical-implementation-with-gridworld-environment-learning-outcomesby-the-end-of-this-exercise-you-will-understand--the-mathematical-foundation-of-mdps--how-to-compute-value-functions--the-relationship-between-policies-and-value-functions--implementation-of-basic-rl-algorithmsdeep-reinforcement-learning---session-2-exercise-markov-decision-processes-and-value-functionsobjective-this-comprehensive-exercise-covers-fundamental-concepts-of-reinforcement-learning-including-markov-decision-processes-mdps-value-functions-bellman-equations-and-policy-evaluation-methods-topics-covered1-introduction-to-reinforcement-learning-framework2-markov-decision-processes-mdps3-value-functions-state-value-and-action-value4-bellman-equations5-policy-evaluation-and-improvement6-practical-implementation-with-gridworld-environment-learning-outcomesby-the-end-of-this-exercise-you-will-understand--the-mathematical-foundation-of-mdps--how-to-compute-value-functions--the-relationship-between-policies-and-value-functions--implementation-of-basic-rl-algorithms--part-1-theoretical-foundation-11-reinforcement-learning-frameworkdefinitionreinforcement-learning-is-a-computational-approach-to-learning-from-interaction-the-key-elements-are--agent-the-learner-and-decision-maker---the-entity-that-makes-choices--environment-the-world-the-agent-interacts-with---everything-outside-the-agent--state-s-current-situation-of-the-agent---describes-the-current-circumstances--action-a-choices-available-to-the-agent---decisions-that-can-be-made--reward-r-numerical-feedback-from-environment---immediate-feedback-signal--policy-π-agents-strategy-for-choosing-actions---mapping-from-states-to-actionsreal-world-analogythink-of-rl-like-learning-to-drive--agent--the-driver-you--environment--roads-traffic-weather-conditions--state--current-speed-position-traffic-around-you--actions--accelerate-brake-turn-leftright--reward--positive-for-safe-driving-negative-for-accidents--policy--your-driving-strategy-cautious-aggressive-etc----12-markov-decision-process-mdpdefinitionan-mdp-is-defined-by-the-tuple-s-a-p-r-γ-where--s-set-of-states---all-possible-situations-the-agent-can-encounter--a-set-of-actions---all-possible-decisions-available-to-the-agent--p-transition-probability-function-pssa---probability-of-moving-to-state-s-given-current-state-s-and-action-a--r-reward-function-rsas---immediate-reward-received-for-transitioning-from-s-to-s-via-action-a--γ-discount-factor-0--γ--1---determines-importance-of-future-rewardsmarkov-propertythe-future-depends-only-on-the-current-state-not-on-the-history-of-how-we-got-there-mathematicallypst1--s--st--s-at--a-st-1-at-1--s0-a0--pst1--s--st--s-at--aintuitionthe-current-state-contains-all-information-needed-to-make-optimal-decisions-the-past-is-already-encoded-in-the-current-state----13-value-functionsstate-value-functionvπs--mathbbeπgt--s_t--sinterpretation-expected-total-reward-when-starting-from-state-s-and-following-policy-π-it-answers-how-good-is-it-to-be-in-this-stateaction-value-functionqπsa--mathbbeπgt--st--s-at--ainterpretation-expected-total-reward-when-taking-action-a-in-state-s-and-then-following-policy-π-it-answers-how-good-is-it-to-take-this-specific-action-in-this-statereturn-cumulative-rewardgt--rt1--γrt2--γ2rt3----sumk0infty-γk-rtk1why-discount-factor-γ--γ--0-only-immediate-rewards-matter-myopic--γ--1-all-future-rewards-are-equally-important--0--γ--1-future-rewards-are-discounted-realistic-for-most-scenarios----14-bellman-equationsbellman-equation-for-state-value-functionvπs--suma-πas-sums-pssarsas--γvπsinterpretation-the-value-of-a-state-equals-the-immediate-reward-plus-the-discounted-value-of-the-next-state-averaged-over-all-possible-actions-and-transitionsbellman-equation-for-action-value-functionqπsa--sums-pssarsas--γ-suma-πasqπsakey-insight-the-bellman-equations-express-a-recursive-relationship---the-value-of-a-state-or-state-action-pair-depends-on-the-immediate-reward-plus-the-discounted-value-of-future-states-this-is-the-mathematical-foundation-for-most-rl-algorithmspart-1-theoretical-foundation-11-reinforcement-learning-frameworkdefinitionreinforcement-learning-is-a-computational-approach-to-learning-from-interaction-the-key-elements-are--agent-the-learner-and-decision-maker---the-entity-that-makes-choices--environment-the-world-the-agent-interacts-with---everything-outside-the-agent--state-s-current-situation-of-the-agent---describes-the-current-circumstances--action-a-choices-available-to-the-agent---decisions-that-can-be-made--reward-r-numerical-feedback-from-environment---immediate-feedback-signal--policy-π-agents-strategy-for-choosing-actions---mapping-from-states-to-actionsreal-world-analogythink-of-rl-like-learning-to-drive--agent--the-driver-you--environment--roads-traffic-weather-conditions--state--current-speed-position-traffic-around-you--actions--accelerate-brake-turn-leftright--reward--positive-for-safe-driving-negative-for-accidents--policy--your-driving-strategy-cautious-aggressive-etc----12-markov-decision-process-mdpdefinitionan-mdp-is-defined-by-the-tuple-s-a-p-r-γ-where--s-set-of-states---all-possible-situations-the-agent-can-encounter--a-set-of-actions---all-possible-decisions-available-to-the-agent--p-transition-probability-function-pssa---probability-of-moving-to-state-s-given-current-state-s-and-action-a--r-reward-function-rsas---immediate-reward-received-for-transitioning-from-s-to-s-via-action-a--γ-discount-factor-0--γ--1---determines-importance-of-future-rewardsmarkov-propertythe-future-depends-only-on-the-current-state-not-on-the-history-of-how-we-got-there-mathematicallypst1--s--st--s-at--a-st-1-at-1--s0-a0--pst1--s--st--s-at--aintuitionthe-current-state-contains-all-information-needed-to-make-optimal-decisions-the-past-is-already-encoded-in-the-current-state----13-value-functionsstate-value-functionvπs--mathbbeπgt--s_t--sinterpretation-expected-total-reward-when-starting-from-state-s-and-following-policy-π-it-answers-how-good-is-it-to-be-in-this-stateaction-value-functionqπsa--mathbbeπgt--st--s-at--ainterpretation-expected-total-reward-when-taking-action-a-in-state-s-and-then-following-policy-π-it-answers-how-good-is-it-to-take-this-specific-action-in-this-statereturn-cumulative-rewardgt--rt1--γrt2--γ2rt3----sumk0infty-γk-rtk1why-discount-factor-γ--γ--0-only-immediate-rewards-matter-myopic--γ--1-all-future-rewards-are-equally-important--0--γ--1-future-rewards-are-discounted-realistic-for-most-scenarios----14-bellman-equationsbellman-equation-for-state-value-functionvπs--suma-πas-sums-pssarsas--γvπsinterpretation-the-value-of-a-state-equals-the-immediate-reward-plus-the-discounted-value-of-the-next-state-averaged-over-all-possible-actions-and-transitionsbellman-equation-for-action-value-functionqπsa--sums-pssarsas--γ-suma-πasqπsakey-insight-the-bellman-equations-express-a-recursive-relationship---the-value-of-a-state-or-state-action-pair-depends-on-the-immediate-reward-plus-the-discounted-value-of-future-states-this-is-the-mathematical-foundation-for-most-rl-algorithms---common-misconceptions-and-clarificationsmisconception-1-the-agent-knows-the-environment-modelreality-in-most-rl-problems-the-agent-doesnt-know-pssa-or-rsas-this-is-called-model-free-rl-where-the-agent-learns-through-trial-and-errormisconception-2-higher-rewards-are-always-betterreality-the-goal-is-to-maximize-cumulative-reward-not-immediate-reward-sometimes-taking-a-small-immediate-reward-prevents-getting-a-much-larger-future-rewardmisconception-3-the-policy-should-always-be-deterministicreality-stochastic-policies-that-output-probabilities-are-often-better-because-they-allow-for-exploration-and-can-be-optimal-in-certain-environments-----building-intuition-restaurant-examplescenario-youre-choosing-restaurants-to-visit-in-a-new-citymdp-components--states-your-hunger-level-location-time-of-day-budget--actions-choose-restaurant-a-b-c-or-cook-at-home--rewards-satisfaction-from-food-immediate--health-effects-long-term--transitions-how-your-state-changes-after-eatingvalue-functions--vhungry-downtown-evening-how-good-is-this-situation-overall--qhungry-downtown-evening-restaurant-a-how-good-is-choosing-restaurant-a-in-this-situationpolicy-learning-initially-random-choices--gradually-prefer-restaurants-that-gave-good-experiences--eventually-develop-a-strategy-that-considers-health-taste-cost-and-convenience-----mathematical-properties-and-theoremstheorem-1-existence-and-uniqueness-of-value-functionsfor-any-policy-π-and-finite-mdp-there-exists-a-unique-solution-to-the-bellman-equationstheorem-2-bellman-optimality-principlea-policy-π-is-optimal-if-and-only-ifvπs--max_a-qπsa-text-for-all--s-in-stheorem-3-policy-improvement-theoremif-π-is-greedy-with-respect-to-vπ-then-vπs--vπs-for-all-states-spractical-implications--we-can-always-improve-a-policy-by-being-greedy-with-respect-to-its-value-function--there-always-exists-an-optimal-policy-may-not-be-unique--the-optimal-value-function-satisfies-the-bellman-optimality-equations--rewards-1-for-safe-driving--10-for-accidents--1-for-speeding-tickets--policy-your-driving-strategy-aggressive-conservative-etc-why-markov-property-mattersthe-markov-property-means-the-future-depends-only-on-the-present-not-the-pastexample-in-chess-to-decide-your-next-move-you-only-need-to-see-the-current-board-position-you-dont-need-to-know-how-the-pieces-got-there---the-complete-game-history-is-irrelevant-for-making-the-optimal-next-movenon-markov-example-predicting-tomorrows-weather-based-only-on-todays-weather-you-need-historical-patterns-understanding-the-discount-factor-γthe-discount-factor-determines-how-much-you-care-about-future-rewards--γ--0-i-only-care-about-immediate-rewards-very-myopic--example-only-caring-about-this-months-salary-not-career-growth---γ--09-future-rewards-are-worth-90-of-immediate-rewards--example-investing-money---you-value-future-returns-but-prefer-sooner---γ--1-future-rewards-are-as-valuable-as-immediate-rewards--example-climate-change-actions---long-term-benefits-matter-equallymathematical-impact--return-gt--rt1--γrt2--γ²rt3----with-γ09-gt--rt1--09rt2--081rt3----future-rewards-get-progressively-less-important-common-misconceptions-and-clarificationsmisconception-1-the-agent-knows-the-environment-modelreality-in-most-rl-problems-the-agent-doesnt-know-pssa-or-rsas-this-is-called-model-free-rl-where-the-agent-learns-through-trial-and-errormisconception-2-higher-rewards-are-always-betterreality-the-goal-is-to-maximize-cumulative-reward-not-immediate-reward-sometimes-taking-a-small-immediate-reward-prevents-getting-a-much-larger-future-rewardmisconception-3-the-policy-should-always-be-deterministicreality-stochastic-policies-that-output-probabilities-are-often-better-because-they-allow-for-exploration-and-can-be-optimal-in-certain-environments-----building-intuition-restaurant-examplescenario-youre-choosing-restaurants-to-visit-in-a-new-citymdp-components--states-your-hunger-level-location-time-of-day-budget--actions-choose-restaurant-a-b-c-or-cook-at-home--rewards-satisfaction-from-food-immediate--health-effects-long-term--transitions-how-your-state-changes-after-eatingvalue-functions--vhungry-downtown-evening-how-good-is-this-situation-overall--qhungry-downtown-evening-restaurant-a-how-good-is-choosing-restaurant-a-in-this-situationpolicy-learning-initially-random-choices--gradually-prefer-restaurants-that-gave-good-experiences--eventually-develop-a-strategy-that-considers-health-taste-cost-and-convenience-----mathematical-properties-and-theoremstheorem-1-existence-and-uniqueness-of-value-functionsfor-any-policy-π-and-finite-mdp-there-exists-a-unique-solution-to-the-bellman-equationstheorem-2-bellman-optimality-principlea-policy-π-is-optimal-if-and-only-ifvπs--max_a-qπsa-text-for-all--s-in-stheorem-3-policy-improvement-theoremif-π-is-greedy-with-respect-to-vπ-then-vπs--vπs-for-all-states-spractical-implications--we-can-always-improve-a-policy-by-being-greedy-with-respect-to-its-value-function--there-always-exists-an-optimal-policy-may-not-be-unique--the-optimal-value-function-satisfies-the-bellman-optimality-equations--rewards-1-for-safe-driving--10-for-accidents--1-for-speeding-tickets--policy-your-driving-strategy-aggressive-conservative-etc-why-markov-property-mattersthe-markov-property-means-the-future-depends-only-on-the-present-not-the-pastexample-in-chess-to-decide-your-next-move-you-only-need-to-see-the-current-board-position-you-dont-need-to-know-how-the-pieces-got-there---the-complete-game-history-is-irrelevant-for-making-the-optimal-next-movenon-markov-example-predicting-tomorrows-weather-based-only-on-todays-weather-you-need-historical-patterns-understanding-the-discount-factor-γthe-discount-factor-determines-how-much-you-care-about-future-rewards--γ--0-i-only-care-about-immediate-rewards-very-myopic--example-only-caring-about-this-months-salary-not-career-growth---γ--09-future-rewards-are-worth-90-of-immediate-rewards--example-investing-money---you-value-future-returns-but-prefer-sooner---γ--1-future-rewards-are-as-valuable-as-immediate-rewards--example-climate-change-actions---long-term-benefits-matter-equallymathematical-impact--return-gt--rt1--γrt2--γ²rt3----with-γ09-gt--rt1--09rt2--081rt3----future-rewards-get-progressively-less-important---understanding-our-gridworld-environmentbefore-we-dive-into-the-code-lets-understand-what-were-building-the-gridworld-setup00----03--x-x---x---30----33-legend--s-at-00-starting-position---at-33-goal-treasure--x-obstacles-walls-or-pits---regular-empty-spaces--arrows-possible-movements-why-this-environment-is-perfect-for-learning1-small--manageable-44-grid--16-states-easy-to-visualize2-clear-objective-get-from-start-to-goal3-interesting-obstacles-forces-strategic-thinking4-deterministic-same-action-always-leads-to-same-result-for-now-reward-structure-explained--goal-reward-10-big-positive-reward-for-reaching-the-treasure--step-penalty--01-small-negative-reward-for-each-move-encourages-efficiency--obstacle-penalty--5-big-negative-reward-for-hitting-obstacles-safety-firstwhy-these-specific-values--goal-reward-is-much-larger-than-step-penalty--encourages-reaching-the-goal--obstacle-penalty-is-significant--discourages-dangerous-moves--step-penalty-is-small--prevents-infinite-wandering-without-being-too-harsh-state-representationeach-state-is-a-tuple-row-column--00--top-left-corner--33--bottom-right-corner---states-are-like-gps-coordinates-for-our-agent-understanding-our-gridworld-environmentbefore-we-dive-into-the-code-lets-understand-what-were-building-the-gridworld-setup00----03--x-x---x---30----33-legend--s-at-00-starting-position---at-33-goal-treasure--x-obstacles-walls-or-pits---regular-empty-spaces--arrows-possible-movements-why-this-environment-is-perfect-for-learning1-small--manageable-44-grid--16-states-easy-to-visualize2-clear-objective-get-from-start-to-goal3-interesting-obstacles-forces-strategic-thinking4-deterministic-same-action-always-leads-to-same-result-for-now-reward-structure-explained--goal-reward-10-big-positive-reward-for-reaching-the-treasure--step-penalty--01-small-negative-reward-for-each-move-encourages-efficiency--obstacle-penalty--5-big-negative-reward-for-hitting-obstacles-safety-firstwhy-these-specific-values--goal-reward-is-much-larger-than-step-penalty--encourages-reaching-the-goal--obstacle-penalty-is-significant--discourages-dangerous-moves--step-penalty-is-small--prevents-infinite-wandering-without-being-too-harsh-state-representationeach-state-is-a-tuple-row-column--00--top-left-corner--33--bottom-right-corner---states-are-like-gps-coordinates-for-our-agent--part-2-policy-definition-and-evaluation-exercise-21-define-different-policiesdefinitiona-policy-πas-defines-the-probability-of-taking-action-a-in-state-s-its-the-agents-strategy-for-choosing-actionsmathematical-representationpias--ptextaction--a--textstate--stypes-of-policies--deterministic-policy-πas--0-1---always-chooses-the-same-action-in-a-given-state--stochastic-policy-πas--0-1---chooses-actions-probabilisticallypolicies-well-implement1-random-policy-equal-probability-for-all-valid-actions2-greedy-policy-always-move-towards-the-goal-3-custom-policy-your-own-strategic-policy----exercise-22-policy-evaluationdefinitionpolicy-evaluation-computes-the-value-function-vπs-for-a-given-policy-π-it-answers-how-good-is-this-policyiterative-policy-evaluation-algorithm1-initialize-vs--0-for-all-states-s2-repeat-until-convergence--for-each-state-s--vnews--σa-πas-σs-pssarsas--γvs3-return-converged-value-function-vconvergence-conditionmaxs-vnews---volds--θ-where-θ-is-a-small-threshold-eg-1e-6intuitionwe-start-with-all-state-values-at-zero-and-iteratively-update-them-based-on-the-bellman-equation-until-they-stabilize-its-like-repeatedly-asking-if-i-follow-this-policy-how-much-reward-will-i-get-until-the-answer-stops-changingpart-2-policy-definition-and-evaluation-exercise-21-define-different-policiesdefinitiona-policy-πas-defines-the-probability-of-taking-action-a-in-state-s-its-the-agents-strategy-for-choosing-actionsmathematical-representationpias--ptextaction--a--textstate--stypes-of-policies--deterministic-policy-πas--0-1---always-chooses-the-same-action-in-a-given-state--stochastic-policy-πas--0-1---chooses-actions-probabilisticallypolicies-well-implement1-random-policy-equal-probability-for-all-valid-actions2-greedy-policy-always-move-towards-the-goal-3-custom-policy-your-own-strategic-policy----exercise-22-policy-evaluationdefinitionpolicy-evaluation-computes-the-value-function-vπs-for-a-given-policy-π-it-answers-how-good-is-this-policyiterative-policy-evaluation-algorithm1-initialize-vs--0-for-all-states-s2-repeat-until-convergence--for-each-state-s--vnews--σa-πas-σs-pssarsas--γvs3-return-converged-value-function-vconvergence-conditionmaxs-vnews---volds--θ-where-θ-is-a-small-threshold-eg-1e-6intuitionwe-start-with-all-state-values-at-zero-and-iteratively-update-them-based-on-the-bellman-equation-until-they-stabilize-its-like-repeatedly-asking-if-i-follow-this-policy-how-much-reward-will-i-get-until-the-answer-stops-changing---policy-deep-dive-understanding-different-strategieswhat-is-a-policya-policy-is-like-a-gps-navigation-system-for-our-agent-it-tells-the-agent-what-to-do-in-every-possible-situationmathematical-definitionπas--probability-of-taking-action-a-when-in-state-s-----types-of-policies-well-implement1-random-policy-strategy-when-in-doubt-flip-a-coinmathematical-definition-πas--1valid_actions-for-all-valid-actionsexample-at-state-10-if-we-can-go-up-down-right-each-has-3333-probabilityadvantages--explores-all-possibilities-equally--simple-to-implement--guarantees-explorationdisadvantages--not-very-efficient--like-wandering-randomly-in-a-maze--no-learning-from-experience---2-greedy-policy-strategy-always-move-closer-to-the-goalmathematical-definition-πas--1-if-a-minimizes-distance-to-goal-0-otherwiseexample-at-state-10-if-goal-is-at-33-prefer-down-and-rightadvantages--very-efficient-when-it-works--direct-path-to-goal--fast-convergencedisadvantages--can-get-stuck-in-local-optima--might-walk-into-obstacles--no-exploration-of-alternative-paths---3-custom-policy-strategy-your-creative-combination-of-strategiesexamples--epsilon-greedy-90-greedy-10-random--safety-first-avoid-actions-that-lead-near-obstacles--wall-follower-stay-close-to-boundaries-----real-world-analogiespolicy-vs-strategy-in-gamesthink-of-different-video-game-playing-styles--aggressive-player-always-attacks-deterministic-policy--defensive-player-always-defends-deterministic-policy--adaptive-player-70-attack-30-defend-stochastic-policywhy-stochastic-policiessometimes-randomness-helps--exploration-discover-new-paths-you-wouldnt-normally-try--unpredictability-in-competitive-games-being-predictable-is-bad--robustness-handle-uncertainty-in-the-environmentrestaurant-choice-analogy--random-policy-pick-restaurants-randomly--greedy-policy-always-go-to-your-current-favorite--epsilon-greedy-policy-usually-go-to-favorite-sometimes-try-something-new-policy-deep-dive-understanding-different-strategieswhat-is-a-policya-policy-is-like-a-gps-navigation-system-for-our-agent-it-tells-the-agent-what-to-do-in-every-possible-situationmathematical-definitionπas--probability-of-taking-action-a-when-in-state-s-----types-of-policies-well-implement1-random-policy-strategy-when-in-doubt-flip-a-coinmathematical-definition-πas--1valid_actions-for-all-valid-actionsexample-at-state-10-if-we-can-go-up-down-right-each-has-3333-probabilityadvantages--explores-all-possibilities-equally--simple-to-implement--guarantees-explorationdisadvantages--not-very-efficient--like-wandering-randomly-in-a-maze--no-learning-from-experience---2-greedy-policy-strategy-always-move-closer-to-the-goalmathematical-definition-πas--1-if-a-minimizes-distance-to-goal-0-otherwiseexample-at-state-10-if-goal-is-at-33-prefer-down-and-rightadvantages--very-efficient-when-it-works--direct-path-to-goal--fast-convergencedisadvantages--can-get-stuck-in-local-optima--might-walk-into-obstacles--no-exploration-of-alternative-paths---3-custom-policy-strategy-your-creative-combination-of-strategiesexamples--epsilon-greedy-90-greedy-10-random--safety-first-avoid-actions-that-lead-near-obstacles--wall-follower-stay-close-to-boundaries-----real-world-analogiespolicy-vs-strategy-in-gamesthink-of-different-video-game-playing-styles--aggressive-player-always-attacks-deterministic-policy--defensive-player-always-defends-deterministic-policy--adaptive-player-70-attack-30-defend-stochastic-policywhy-stochastic-policiessometimes-randomness-helps--exploration-discover-new-paths-you-wouldnt-normally-try--unpredictability-in-competitive-games-being-predictable-is-bad--robustness-handle-uncertainty-in-the-environmentrestaurant-choice-analogy--random-policy-pick-restaurants-randomly--greedy-policy-always-go-to-your-current-favorite--epsilon-greedy-policy-usually-go-to-favorite-sometimes-try-something-new---understanding-policy-evaluation-step-by-steppolicy-evaluation-answers-the-question-how-good-is-each-state-if-i-follow-this-policy-the-intuitionimagine-youre-evaluating-different-starting-positions-in-a-board-game--some-positions-are-naturally-better-closer-to-winning--some-positions-are-worse-closer-to-losing---the-value-of-a-position-depends-on-how-well-youll-do-from-there-mathematical-breakdownthe-bellman-equation-for-state-valuesvπs--σa-πas--σs-pssa--rsas--γ--vπslets-decode-this-step-by-step1-for-each-possible-action-a-πas--how-likely-am-i-to-take-action-a-in-state-s2-for-each-possible-next-state-s-pssa--if-i-take-action-a-whats-the-chance-i-end-up-in-state-s3-calculate-immediate-reward--future-value-rsas--γ--vπs--rsas--what-reward-do-i-get-immediately--γ--vπs--whats-the-discounted-future-value4-sum-everything-up-this-gives-the-expected-value-of-being-in-state-s-simple-examplelets-say-were-at-state-22-with-a-random-policypython-random-policy-equal-probability-for-all-valid-actionsπups--025-πdowns--025-πlefts--025-πrights--025-for-action-up--next-state-12contributionup--025--10---01--09--v12-for-action-down--next-state-32contributiondown--025--10---01--09--v32--and-so-on-for-left-and-rightv22--contributionup--contributiondown--contributionleft--contributionright-why-iterative--we-start-with-vs--0-for-all-states-initial-guess--each-iteration-improves-our-estimate-using-current-values--eventually-values-converge-to-true-values--like-asking-if-i-knew-the-value-of-my-neighbors-what-would-my-value-be-convergence-intuitionthink-of-it-like-gossip-spreading-in-a-neighborhood--initially-nobody-knows-the-true-gossip-values--each-iteration-neighbors-share-information---eventually-everyone-converges-to-the-same-true-story-understanding-policy-evaluation-step-by-steppolicy-evaluation-answers-the-question-how-good-is-each-state-if-i-follow-this-policy-the-intuitionimagine-youre-evaluating-different-starting-positions-in-a-board-game--some-positions-are-naturally-better-closer-to-winning--some-positions-are-worse-closer-to-losing---the-value-of-a-position-depends-on-how-well-youll-do-from-there-mathematical-breakdownthe-bellman-equation-for-state-valuesvπs--σa-πas--σs-pssa--rsas--γ--vπslets-decode-this-step-by-step1-for-each-possible-action-a-πas--how-likely-am-i-to-take-action-a-in-state-s2-for-each-possible-next-state-s-pssa--if-i-take-action-a-whats-the-chance-i-end-up-in-state-s3-calculate-immediate-reward--future-value-rsas--γ--vπs--rsas--what-reward-do-i-get-immediately--γ--vπs--whats-the-discounted-future-value4-sum-everything-up-this-gives-the-expected-value-of-being-in-state-s-simple-examplelets-say-were-at-state-22-with-a-random-policypython-random-policy-equal-probability-for-all-valid-actionsπups--025-πdowns--025-πlefts--025-πrights--025-for-action-up--next-state-12contributionup--025--10---01--09--v12-for-action-down--next-state-32contributiondown--025--10---01--09--v32--and-so-on-for-left-and-rightv22--contributionup--contributiondown--contributionleft--contributionright-why-iterative--we-start-with-vs--0-for-all-states-initial-guess--each-iteration-improves-our-estimate-using-current-values--eventually-values-converge-to-true-values--like-asking-if-i-knew-the-value-of-my-neighbors-what-would-my-value-be-convergence-intuitionthink-of-it-like-gossip-spreading-in-a-neighborhood--initially-nobody-knows-the-true-gossip-values--each-iteration-neighbors-share-information---eventually-everyone-converges-to-the-same-true-story--exercise-23-create-your-custom-policytask-design-and-implement-your-own-policy-consider-strategies-like--wall-following-try-to-stay-close-to-walls--risk-averse-avoid-obstacles-with-higher-probability--exploration-focused-balance-between-moving-towards-goal-and-exploringyour-implementation-belowexercise-23-create-your-custom-policytask-design-and-implement-your-own-policy-consider-strategies-like--wall-following-try-to-stay-close-to-walls--risk-averse-avoid-obstacles-with-higher-probability--exploration-focused-balance-between-moving-towards-goal-and-exploringyour-implementation-below--part-3-action-value-functions-q-functions-exercise-31-computing-q-valuesdefinitionthe-action-value-function-qπsa-represents-the-expected-return-when-taking-action-a-in-state-s-and-then-following-policy-πkey-question-q-functions-answerq-functions-answer-what-if-i-take-this-specific-action-here-then-follow-my-policymathematical-relationshipsv-from-q-policy-weighted-averagevπs--suma-πas-qπsaq-from-v-bellman-backupqπsa--sums-pssarsas--γvπsbellman-equation-for-qqπsa--sums-pssarsas--γ-suma-πasqπsaintuition--vs-how-good-is-this-state-following-current-policy--qsa-how-good-is-this-specific-action-then-following-policythe-v-q-relationship-is-like-asking--v-how-well-will-i-do-from-this-chess-position--q-how-well-will-i-do-if-i-move-my-queen-here-then-play-normallypart-3-action-value-functions-q-functions-exercise-31-computing-q-valuesdefinitionthe-action-value-function-qπsa-represents-the-expected-return-when-taking-action-a-in-state-s-and-then-following-policy-πkey-question-q-functions-answerq-functions-answer-what-if-i-take-this-specific-action-here-then-follow-my-policymathematical-relationshipsv-from-q-policy-weighted-averagevπs--suma-πas-qπsaq-from-v-bellman-backupqπsa--sums-pssarsas--γvπsbellman-equation-for-qqπsa--sums-pssarsas--γ-suma-πasqπsaintuition--vs-how-good-is-this-state-following-current-policy--qsa-how-good-is-this-specific-action-then-following-policythe-v-q-relationship-is-like-asking--v-how-well-will-i-do-from-this-chess-position--q-how-well-will-i-do-if-i-move-my-queen-here-then-play-normally---q-functions-deep-dive-the-what-if-valuescore-conceptq-functions-provide-action-specific-evaluations-allowing-us-to-compare-different-choices-directly-----restaurant-decision-analogyscenario-youre-choosing-a-restaurant-from-downtown-locationvalue-functions--vdowntown--75--average-satisfaction-from-this-location-with-my-usual-choices--qdowntown-pizzaplace--82--satisfaction-if-i-specifically-choose-pizza--qdowntown-sushiplace--68--satisfaction-if-i-specifically-choose-sushi--qdowntown-burgerplace--71--satisfaction-if-i-specifically-choose-burgerspolicy-calculationif-my-policy-is-50-pizza-30-sushi-20-burgersvdowntown--0582--0368--0271--41--204--142--756------mathematical-relationships-explained1-v-from-q-weighted-averagevπs--suma-πas--qπsainterpretation-state-value--probability-of-each-action--value-of-that-action2-q-from-v-bellman-backupqπsa--sums-pssa--rsas--γvπsinterpretation-action-value--immediate-reward--discounted-future-state-value-----why-q-functions-matter1-direct-action-comparison--qs-left--52-vs-qs-right--78--choose-right--no-need-to-compute-state-values-first2-policy-improvement--πnews--argmaxa-qπoldsa--directly-find-the-best-action3-optimal-decision-making--qsa-tells-us-the-value-of-each-action-under-optimal-behavior--essential-for-q-learning-algorithms-----visual-understandingthink-of-q-values-as-action-specific-heat-maps--hot-spots-high-q-values-good-actions-to-take--cold-spots-low-q-values-actions-to-avoid--separate-map-for-each-action-qs-qs-qs-qsgridworld-example--qstate-towardgoal-typically-has-higher-values--qstate-towardobstacle-typically-has-lower-values--qstate-towardwall-often-has-negative-values---like-restaurant-satisfaction--meal-quality--how-ill-feel-tomorrow-why-q-functions-matter1-better-decision-making-q-values-directly-tell-us-which-action-is-best--maxa-qsa-gives-the-best-action-in-state-s2-policy-improvement-we-can-improve-policies-by-being-greedy-wrt-q-values--πnews--argmaxa-qπ_oldsa3-action-comparison-compare-different-actions-in-the-same-state--should-i-go-left-or-right-from-here-visual-understandingthink-of-q-values-as-a-heat-map-for-each-action--hot-spots-high-q-values-good-actions-to-take--cold-spots-low-q-values-actions-to-avoid---different-maps-for-each-action-qsup-qsdown-qsleft-qsright-common-confusion-v-vs-q--vs-how-good-is-my-current-strategy-from-this-position--qsa-how-good-is-this-specific-move-then-using-my-strategyits-like-asking--v-how-well-will-i-do-in-this-chess-position---q-how-well-will-i-do-if-i-move-my-queen-here-then-play-normally-q-functions-deep-dive-the-what-if-valuescore-conceptq-functions-provide-action-specific-evaluations-allowing-us-to-compare-different-choices-directly-----restaurant-decision-analogyscenario-youre-choosing-a-restaurant-from-downtown-locationvalue-functions--vdowntown--75--average-satisfaction-from-this-location-with-my-usual-choices--qdowntown-pizzaplace--82--satisfaction-if-i-specifically-choose-pizza--qdowntown-sushiplace--68--satisfaction-if-i-specifically-choose-sushi--qdowntown-burgerplace--71--satisfaction-if-i-specifically-choose-burgerspolicy-calculationif-my-policy-is-50-pizza-30-sushi-20-burgersvdowntown--0582--0368--0271--41--204--142--756------mathematical-relationships-explained1-v-from-q-weighted-averagevπs--suma-πas--qπsainterpretation-state-value--probability-of-each-action--value-of-that-action2-q-from-v-bellman-backupqπsa--sums-pssa--rsas--γvπsinterpretation-action-value--immediate-reward--discounted-future-state-value-----why-q-functions-matter1-direct-action-comparison--qs-left--52-vs-qs-right--78--choose-right--no-need-to-compute-state-values-first2-policy-improvement--πnews--argmaxa-qπoldsa--directly-find-the-best-action3-optimal-decision-making--qsa-tells-us-the-value-of-each-action-under-optimal-behavior--essential-for-q-learning-algorithms-----visual-understandingthink-of-q-values-as-action-specific-heat-maps--hot-spots-high-q-values-good-actions-to-take--cold-spots-low-q-values-actions-to-avoid--separate-map-for-each-action-qs-qs-qs-qsgridworld-example--qstate-towardgoal-typically-has-higher-values--qstate-towardobstacle-typically-has-lower-values--qstate-towardwall-often-has-negative-values---like-restaurant-satisfaction--meal-quality--how-ill-feel-tomorrow-why-q-functions-matter1-better-decision-making-q-values-directly-tell-us-which-action-is-best--maxa-qsa-gives-the-best-action-in-state-s2-policy-improvement-we-can-improve-policies-by-being-greedy-wrt-q-values--πnews--argmaxa-qπ_oldsa3-action-comparison-compare-different-actions-in-the-same-state--should-i-go-left-or-right-from-here-visual-understandingthink-of-q-values-as-a-heat-map-for-each-action--hot-spots-high-q-values-good-actions-to-take--cold-spots-low-q-values-actions-to-avoid---different-maps-for-each-action-qsup-qsdown-qsleft-qsright-common-confusion-v-vs-q--vs-how-good-is-my-current-strategy-from-this-position--qsa-how-good-is-this-specific-move-then-using-my-strategyits-like-asking--v-how-well-will-i-do-in-this-chess-position---q-how-well-will-i-do-if-i-move-my-queen-here-then-play-normally--part-4-policy-improvement-and-policy-iteration-exercise-41-policy-improvementdefinitiongiven-a-value-function-vπ-we-can-improve-the-policy-by-being-greedy-with-respect-to-the-action-value-functionpolicy-improvement-formulaπs--argmaxa-qπsa--argmaxa-sums-pssarsas--γvπsinterpretation-choose-the-action-that-maximizes-expected-return-from-each-statepolicy-improvement-theoremif-π-is-greedy-with-respect-to-vπ-then-vπs--vπs-for-all-states-stranslation-if-i-always-choose-the-best-action-based-on-my-current-understanding-i-can-only-do-better-or-at-least-as-well----exercise-42-policy-iteration-algorithmpolicy-iteration-steps1-initialize-start-with-arbitrary-policy-π₀2-repeat-until-convergence--policy-evaluation-compute-vπk-solve-bellman-equation--policy-improvement-πk1s--argmaxa-qπ_ksa3-output-optimal-policy-π-and-value-function-vconvergence-guarantee-policy-iteration-is-guaranteed-to-converge-to-the-optimal-policy-in-finite-time-for-finite-mdpswhy-it-works--each-step-produces-a-better-or-equal-policy--there-are-only-finitely-many-deterministic-policies--must-eventually-reach-optimal-policypart-4-policy-improvement-and-policy-iteration-exercise-41-policy-improvementdefinitiongiven-a-value-function-vπ-we-can-improve-the-policy-by-being-greedy-with-respect-to-the-action-value-functionpolicy-improvement-formulaπs--argmaxa-qπsa--argmaxa-sums-pssarsas--γvπsinterpretation-choose-the-action-that-maximizes-expected-return-from-each-statepolicy-improvement-theoremif-π-is-greedy-with-respect-to-vπ-then-vπs--vπs-for-all-states-stranslation-if-i-always-choose-the-best-action-based-on-my-current-understanding-i-can-only-do-better-or-at-least-as-well----exercise-42-policy-iteration-algorithmpolicy-iteration-steps1-initialize-start-with-arbitrary-policy-π₀2-repeat-until-convergence--policy-evaluation-compute-vπk-solve-bellman-equation--policy-improvement-πk1s--argmaxa-qπ_ksa3-output-optimal-policy-π-and-value-function-vconvergence-guarantee-policy-iteration-is-guaranteed-to-converge-to-the-optimal-policy-in-finite-time-for-finite-mdpswhy-it-works--each-step-produces-a-better-or-equal-policy--there-are-only-finitely-many-deterministic-policies--must-eventually-reach-optimal-policy---policy-improvement-deep-dive-making-better-decisionscore-idea-use-the-value-function-to-make-better-action-choices-----learning-process-analogyscenario-youre-learning-to-play-chesspolicy-evaluation-how-good-is-my-current-playing-style--analyze-your-current-strategy--evaluate-typical-game-outcomes--identify-strengths-and-weaknessespolicy-improvement-how-can-i-play-better--look-at-each-position-where-you-made-suboptimal-moves--replace-bad-moves-with-better-alternatives--update-your-playing-strategypolicy-iteration-repeat-this-cycle-until-you-cant-improve-further-----mathematical-foundationspolicy-improvement-theoremif-π-is-greedy-wrt-vπ-then-vπs--vπs-for-all-sproof-intuition1-greedy-action-choose-a-such-that-qπsa-is-maximized2-definition-qπsa--vπs-for-the-chosen-action3-new-policy-πs-gives-this-optimal-action4-result-vπs--vπswhy-greedy-improvement-works--current-policy-chooses-actions-with-average-value-vπs--greedy-policy-chooses-action-with-maximum-value-qπsa--maximum--average-so-new-policy-is-better-----policy-iteration-the-complete-cyclestep-1---policy-evaluation-how-good-is-my-current-policyvπs--expected-return-following-π-from-state-sstep-2---policy-improvement-whats-the-best-action-in-each-stateπs--action-that-maximizes-qπsastep-3---check-convergence-did-my-policy-changeif-πs--πs-for-all-s-stop-optimal-foundelse-π--π-and-repeat-----key-properties1-monotonic-improvementvπ₀--vπ₁--vπ₂----vπ2-finite-convergencealgorithm-terminates-in-finite-steps-for-finite-mdps3-optimal-solutionfinal-policy-π-is-optimal-vπ--v4-model-basedrequires-knowledge-of-transition-probabilities-pssa-and-rewards-rsasthink-of-a-student-improving-their-study-strategy1-current-strategy-policy-π-i-study-randomly-for-2-hours2-evaluate-strategy-policy-evaluation-how-well-does-this-work-for-each-subject-3-find-better-strategy-policy-improvement-math-needs-3-hours-history-needs-1-hour4-repeat-keep-refining-until-no-more-improvements-possible-mathematical-intuitionpolicy-improvement-theorem-if-qπsa--vπs-for-some-action-a-then-taking-action-a-is-better-than-following-policy-πtranslation-if-doing-action-a-gives-higher-value-than-my-current-average-i-should-do-action-a-more-oftengreedy-improvementpythonπnews--argmaxa-qπsaalways-choose-the-action-with-highest-q-value-why-does-this-workmonotonic-improvement-each-policy-improvement-step-makes-the-policy-at-least-as-good-usually-betterproof-sketch--if-were-greedy-wrt-qπ-we-get-vπ_new--vπ--if-i-always-choose-the-best-available-action-i-cant-do-worse-policy-iteration-the-complete-algorithmthe-cyclerandom-policy--evaluate--improve--evaluate--improve----optimal-policywhy-it-converges1-finite-stateaction-space-limited-number-of-possible-policies2-monotonic-improvement-each-step-makes-policy-better-or-same3-no-cycles-cant-go-backwards-to-a-worse-policy4-must-terminate-eventually-reach-optimal-policy-real-world-example-learning-to-driveiteration-1--policy-drive-slowly-everywhere---evaluation-safe-but-inefficient-on-highways--improvement-drive-fast-on-highways-slow-in-neighborhoodsiteration-2--policy-speed-varies-by-road-type--evaluation-good-but-inefficient-in-traffic---improvement-also-consider-traffic-conditionsfinal-policy-optimal-speed-based-on-road-type-traffic-weather-etc-key-insights1-guaranteed-improvement-policy-iteration-always-finds-the-optimal-policy-for-finite-mdps2-fast-convergence-usually-converges-in-just-a-few-iterations3-no-exploration-needed-uses-complete-model-knowledge-unlike-q-learning-later4-computational-cost-each-iteration-requires-solving-the-bellman-equation-common-pitfalls--getting-stuck-in-stochastic-environments-might-need-exploration--computational-cost-policy-evaluation-can-be-expensive---model-required-need-to-know-pssa-and-rsas-policy-improvement-deep-dive-making-better-decisionscore-idea-use-the-value-function-to-make-better-action-choices-----learning-process-analogyscenario-youre-learning-to-play-chesspolicy-evaluation-how-good-is-my-current-playing-style--analyze-your-current-strategy--evaluate-typical-game-outcomes--identify-strengths-and-weaknessespolicy-improvement-how-can-i-play-better--look-at-each-position-where-you-made-suboptimal-moves--replace-bad-moves-with-better-alternatives--update-your-playing-strategypolicy-iteration-repeat-this-cycle-until-you-cant-improve-further-----mathematical-foundationspolicy-improvement-theoremif-π-is-greedy-wrt-vπ-then-vπs--vπs-for-all-sproof-intuition1-greedy-action-choose-a-such-that-qπsa-is-maximized2-definition-qπsa--vπs-for-the-chosen-action3-new-policy-πs-gives-this-optimal-action4-result-vπs--vπswhy-greedy-improvement-works--current-policy-chooses-actions-with-average-value-vπs--greedy-policy-chooses-action-with-maximum-value-qπsa--maximum--average-so-new-policy-is-better-----policy-iteration-the-complete-cyclestep-1---policy-evaluation-how-good-is-my-current-policyvπs--expected-return-following-π-from-state-sstep-2---policy-improvement-whats-the-best-action-in-each-stateπs--action-that-maximizes-qπsastep-3---check-convergence-did-my-policy-changeif-πs--πs-for-all-s-stop-optimal-foundelse-π--π-and-repeat-----key-properties1-monotonic-improvementvπ₀--vπ₁--vπ₂----vπ2-finite-convergencealgorithm-terminates-in-finite-steps-for-finite-mdps3-optimal-solutionfinal-policy-π-is-optimal-vπ--v4-model-basedrequires-knowledge-of-transition-probabilities-pssa-and-rewards-rsasthink-of-a-student-improving-their-study-strategy1-current-strategy-policy-π-i-study-randomly-for-2-hours2-evaluate-strategy-policy-evaluation-how-well-does-this-work-for-each-subject-3-find-better-strategy-policy-improvement-math-needs-3-hours-history-needs-1-hour4-repeat-keep-refining-until-no-more-improvements-possible-mathematical-intuitionpolicy-improvement-theorem-if-qπsa--vπs-for-some-action-a-then-taking-action-a-is-better-than-following-policy-πtranslation-if-doing-action-a-gives-higher-value-than-my-current-average-i-should-do-action-a-more-oftengreedy-improvementpythonπnews--argmaxa-qπsaalways-choose-the-action-with-highest-q-value-why-does-this-workmonotonic-improvement-each-policy-improvement-step-makes-the-policy-at-least-as-good-usually-betterproof-sketch--if-were-greedy-wrt-qπ-we-get-vπ_new--vπ--if-i-always-choose-the-best-available-action-i-cant-do-worse-policy-iteration-the-complete-algorithmthe-cyclerandom-policy--evaluate--improve--evaluate--improve----optimal-policywhy-it-converges1-finite-stateaction-space-limited-number-of-possible-policies2-monotonic-improvement-each-step-makes-policy-better-or-same3-no-cycles-cant-go-backwards-to-a-worse-policy4-must-terminate-eventually-reach-optimal-policy-real-world-example-learning-to-driveiteration-1--policy-drive-slowly-everywhere---evaluation-safe-but-inefficient-on-highways--improvement-drive-fast-on-highways-slow-in-neighborhoodsiteration-2--policy-speed-varies-by-road-type--evaluation-good-but-inefficient-in-traffic---improvement-also-consider-traffic-conditionsfinal-policy-optimal-speed-based-on-road-type-traffic-weather-etc-key-insights1-guaranteed-improvement-policy-iteration-always-finds-the-optimal-policy-for-finite-mdps2-fast-convergence-usually-converges-in-just-a-few-iterations3-no-exploration-needed-uses-complete-model-knowledge-unlike-q-learning-later4-computational-cost-each-iteration-requires-solving-the-bellman-equation-common-pitfalls--getting-stuck-in-stochastic-environments-might-need-exploration--computational-cost-policy-evaluation-can-be-expensive---model-required-need-to-know-pssa-and-rsas--part-5-experiments-and-analysis-exercise-51-effect-of-discount-factor-γdefinitionthe-discount-factor-γ-determines-how-much-we-value-future-rewards-compared-to-immediate-rewardsmathematical-impactgt--rt1--γrt2--γ2rt3----sumk0infty-γk-rtk1interpretation-of-different-values--γ--0-only-immediate-rewards-matter-myopic-behavior--γ--1-all-future-rewards-equally-important-infinite-horizon--0--γ--1-future-rewards-are-discounted-realistictask-experiment-with-different-discount-factors-and-analyze-their-effect-on-the-optimal-policyresearch-questions1-how-does-γ-affect-the-optimal-policy2-which-γ-values-lead-to-faster-convergence3-what-happens-to-state-values-as-γ-changespart-5-experiments-and-analysis-exercise-51-effect-of-discount-factor-γdefinitionthe-discount-factor-γ-determines-how-much-we-value-future-rewards-compared-to-immediate-rewardsmathematical-impactgt--rt1--γrt2--γ2rt3----sumk0infty-γk-rtk1interpretation-of-different-values--γ--0-only-immediate-rewards-matter-myopic-behavior--γ--1-all-future-rewards-equally-important-infinite-horizon--0--γ--1-future-rewards-are-discounted-realistictask-experiment-with-different-discount-factors-and-analyze-their-effect-on-the-optimal-policyresearch-questions1-how-does-γ-affect-the-optimal-policy2-which-γ-values-lead-to-faster-convergence3-what-happens-to-state-values-as-γ-changes---discount-factor-deep-dive-balancing-present-vs-futurecore-concept-the-discount-factor-γ-controls-the-agents-patience-or-time-preference-----time-value-of-rewardsfinancial-analogyjust-like-money-rewards-have-time-value--100-today-vs-100-in-10-years--most-prefer-today-inflation-uncertainty--10-reward-now-vs-10-reward-in-100-time-steps--usually-prefer-immediatemathematical-effect--γ--01-reward-10-steps-away-is-worth-01¹⁰--00000000001-of-current-reward--γ--09-reward-10-steps-away-is-worth-09¹⁰--035-of-current-reward--γ--099-reward-10-steps-away-is-worth-099¹⁰--090-of-current-reward-----real-world-analogiesγ--01-very-impatientmyopic---i-want-pizza-now-dont-care-about-health-consequences---buy-with-credit-card-ignore-interest-charges---take-fastest-route-ignore-traffic-finesγ--05-moderately-patient---exercise-sometimes-for-health-benefits---save-some-money-spend-some-now---study-when-motivated-party-when-notγ--09-balanced---exercise-regularly-for-long-term-health---study-hard-now-for-career-benefits-later---invest-consistently-for-retirementγ--099-very-patient---plant-trees-for-future-generations---buy-house-as-long-term-investment---address-climate-change-for-distant-future-----effect-on-optimal-policylow-γ-myopic-behavior--takes-shortest-immediate-path-to-reward--ignores-long-term-consequences--may-get-stuck-in-local-optima--fast-convergence-but-potentially-poor-solutionshigh-γ-farsighted-behavior--considers-long-term-consequences--may-take-longer-paths-for-better-future-outcomes--explores-more-thoroughly--slower-convergence-but-better-final-solutionsin-gridworld-context--low-γ-rushes-toward-goal-ignoring-obstacles--high-γ-carefully-plans-path-avoids-risky-moves-mathematical-impactreturn-formula-gt--rt1--γrt2--γ²rt3--γ³rt4--examplesγ--09-patient-agent--gt--rt1--09rt2--081rt3--0729rt4----reward-in-1-step-worth-100-of-immediate-reward--reward-in-2-steps-worth-90-of-immediate-reward---reward-in-3-steps-worth-81-of-immediate-reward--reward-in-10-steps-worth-35-of-immediate-rewardγ--01-impatient-agent--gt--rt1--01rt2--001rt3--0001r_t4----reward-in-2-steps-worth-only-10-of-immediate-reward--reward-in-3-steps-worth-only-1-of-immediate-reward--very-myopic---only-cares-about-next-few-steps-real-world-analogiesγ--01-very-impatient---i-want-pizza-now-dont-care-about-health-consequences---buy-the-cheapest-phone-ignore-long-term-durability----take-the-fastest-route-ignore-traffic-finesγ--09-balanced---exercise-now-for-health-benefits-later---study-hard-now-for-career-benefits-later---invest-money-for-retirementγ--099-very-patient---plant-trees-for-future-generations---buy-a-house-as-long-term-investment---address-climate-change-for-distant-future-effect-on-optimal-policylow-γ-myopic-behavior--takes-shortest-path-to-goal--ignores-long-term-consequences---might-take-dangerous-shortcuts--policy-rush-to-goal-avoid-obstacles-minimallyhigh-γ-farsighted-behavior--takes-safer-longer-paths--values-long-term-safety--more-conservative-decisions--policy-get-to-goal-safely-even-if-it-takes-longer-choosing-γ-in-practiceconsider1-problem-horizon-short-term-tasks--lower-γ-long-term-tasks--higher-γ2-uncertainty-more-uncertain-future--lower-γ3-safety-safety-critical-applications--higher-γ4-computational-higher-γ--slower-convergencecommon-values--γ--09-general-purpose-good-balance--γ--095-099-long-term-planning-tasks--γ--01-05-short-term-reactive-tasks--γ--10-infinite-horizon-theoretical-studies-can-cause-issues-debugging-with-γif-your-agent--ignores-long-term-rewards-increase-γ--is-too-conservative-decrease-γ---wont-converge-check-if-γ-is-too-close-to-1--makes-random-decisions-γ-might-be-too-low-discount-factor-deep-dive-balancing-present-vs-futurecore-concept-the-discount-factor-γ-controls-the-agents-patience-or-time-preference-----time-value-of-rewardsfinancial-analogyjust-like-money-rewards-have-time-value--100-today-vs-100-in-10-years--most-prefer-today-inflation-uncertainty--10-reward-now-vs-10-reward-in-100-time-steps--usually-prefer-immediatemathematical-effect--γ--01-reward-10-steps-away-is-worth-01¹⁰--00000000001-of-current-reward--γ--09-reward-10-steps-away-is-worth-09¹⁰--035-of-current-reward--γ--099-reward-10-steps-away-is-worth-099¹⁰--090-of-current-reward-----real-world-analogiesγ--01-very-impatientmyopic---i-want-pizza-now-dont-care-about-health-consequences---buy-with-credit-card-ignore-interest-charges---take-fastest-route-ignore-traffic-finesγ--05-moderately-patient---exercise-sometimes-for-health-benefits---save-some-money-spend-some-now---study-when-motivated-party-when-notγ--09-balanced---exercise-regularly-for-long-term-health---study-hard-now-for-career-benefits-later---invest-consistently-for-retirementγ--099-very-patient---plant-trees-for-future-generations---buy-house-as-long-term-investment---address-climate-change-for-distant-future-----effect-on-optimal-policylow-γ-myopic-behavior--takes-shortest-immediate-path-to-reward--ignores-long-term-consequences--may-get-stuck-in-local-optima--fast-convergence-but-potentially-poor-solutionshigh-γ-farsighted-behavior--considers-long-term-consequences--may-take-longer-paths-for-better-future-outcomes--explores-more-thoroughly--slower-convergence-but-better-final-solutionsin-gridworld-context--low-γ-rushes-toward-goal-ignoring-obstacles--high-γ-carefully-plans-path-avoids-risky-moves-mathematical-impactreturn-formula-gt--rt1--γrt2--γ²rt3--γ³rt4--examplesγ--09-patient-agent--gt--rt1--09rt2--081rt3--0729rt4----reward-in-1-step-worth-100-of-immediate-reward--reward-in-2-steps-worth-90-of-immediate-reward---reward-in-3-steps-worth-81-of-immediate-reward--reward-in-10-steps-worth-35-of-immediate-rewardγ--01-impatient-agent--gt--rt1--01rt2--001rt3--0001r_t4----reward-in-2-steps-worth-only-10-of-immediate-reward--reward-in-3-steps-worth-only-1-of-immediate-reward--very-myopic---only-cares-about-next-few-steps-real-world-analogiesγ--01-very-impatient---i-want-pizza-now-dont-care-about-health-consequences---buy-the-cheapest-phone-ignore-long-term-durability----take-the-fastest-route-ignore-traffic-finesγ--09-balanced---exercise-now-for-health-benefits-later---study-hard-now-for-career-benefits-later---invest-money-for-retirementγ--099-very-patient---plant-trees-for-future-generations---buy-a-house-as-long-term-investment---address-climate-change-for-distant-future-effect-on-optimal-policylow-γ-myopic-behavior--takes-shortest-path-to-goal--ignores-long-term-consequences---might-take-dangerous-shortcuts--policy-rush-to-goal-avoid-obstacles-minimallyhigh-γ-farsighted-behavior--takes-safer-longer-paths--values-long-term-safety--more-conservative-decisions--policy-get-to-goal-safely-even-if-it-takes-longer-choosing-γ-in-practiceconsider1-problem-horizon-short-term-tasks--lower-γ-long-term-tasks--higher-γ2-uncertainty-more-uncertain-future--lower-γ3-safety-safety-critical-applications--higher-γ4-computational-higher-γ--slower-convergencecommon-values--γ--09-general-purpose-good-balance--γ--095-099-long-term-planning-tasks--γ--01-05-short-term-reactive-tasks--γ--10-infinite-horizon-theoretical-studies-can-cause-issues-debugging-with-γif-your-agent--ignores-long-term-rewards-increase-γ--is-too-conservative-decrease-γ---wont-converge-check-if-γ-is-too-close-to-1--makes-random-decisions-γ-might-be-too-low--exercise-52-modified-environment-experimentstask-a-modify-the-reward-structure-and-analyze-how-it-affects-the-optimal-policy--change-step-reward-from--01-to--10-higher-cost-for-each-step--change-goal-reward-from-10-to-5--add-positive-rewards-for-certain-statestask-b-experiment-with-different-obstacle-configurations--remove-some-obstacles--add-more-obstacles--change-obstacle-positionstask-c-test-with-different-starting-positions-and-analyze-convergenceexercise-52-modified-environment-experimentstask-a-modify-the-reward-structure-and-analyze-how-it-affects-the-optimal-policy--change-step-reward-from--01-to--10-higher-cost-for-each-step--change-goal-reward-from-10-to-5--add-positive-rewards-for-certain-statestask-b-experiment-with-different-obstacle-configurations--remove-some-obstacles--add-more-obstacles--change-obstacle-positionstask-c-test-with-different-starting-positions-and-analyze-convergence--part-6-summary-and-key-takeaways-what-weve-learned1-markov-decision-processes-mdps--framework-sequential-decision-making-under-uncertainty--components-s-a-p-r-γ---states-actions-transitions-rewards-discount--markov-property-future-depends-only-on-current-state-not-history--foundation-mathematical-basis-for-all-rl-algorithms2-value-functions--vπs-expected-return-starting-from-state-s-following-policy-π---qπsa-expected-return-taking-action-a-in-state-s-then-following-π--relationship-vπs--σa-πas-qπsa--purpose-measure-goodness-of-states-and-actions3-bellman-equations--for-v-vπs--σa-πas-σs-pssarsas--γvπs--for-q-qπsa--σs-pssarsas--γ-σa-πasqπsa--significance-recursive-relationship-enabling-dynamic-programming-solutions4-policy-evaluation--algorithm-iterative-method-to-compute-vπ-given-policy-π--convergence-guaranteed-for-finite-mdps-with-γ--1--application-foundation-for-policy-iteration-and-value-iteration5-policy-improvement--theorem-greedy-policy-wrt-vπ-is-at-least-as-good-as-π--formula-πs--argmaxa-qπsa--monotonicity-each-improvement-step-yields-better-or-equal-policy6-policy-iteration--algorithm-alternates-between-evaluation-and-improvement--guarantee-converges-to-optimal-policy-π--efficiency-usually-converges-in-few-iterations----key-insights-from-experimentsdiscount-factor-γ-effects--low-γ-myopic-behavior-focuses-on-immediate-rewards--high-γ-farsighted-behavior-considers-long-term-consequences--trade-off-convergence-speed-vs-solution-qualityenvironment-structure-impact--reward-structure-significantly-affects-optimal-policy--obstacles-create-navigation-challenges-requiring-planning--starting-position-can-influence-learning-dynamicsalgorithm-characteristics--model-based-requires-knowledge-of-pssa-and-rsas--exact-solution-finds-truly-optimal-policy-unlike-approximate-methods--computational-cost-scales-with-state-space-size----connections-to-advanced-topicswhat-this-enables--value-iteration-direct-optimization-of-value-function--q-learning-model-free-learning-of-action-value-functions--deep-rl-neural-network-function-approximation--policy-gradients-direct-policy-optimization-methodsnext-steps-in-learning--temporal-difference-learning-learn-from-incomplete-episodes--function-approximation-handle-largecontinuous-state-spaces--exploration-vs-exploitation-balance-learning-and-performance--multi-agent-systems-multiple-learning-agents-interacting----reflection-questionstheoretical-understanding1-how-would-stochastic-transitions-affect-the-optimal-policy2-what-happens-with-continuous-state-or-action-spaces3-how-do-we-handle-unknown-environment-dynamics4-what-are-computational-limits-for-large-state-spacespractical-applications1-how-could-you-apply-mdps-to-real-world-decision-problems2-what-modifications-would-be-needed-for-competitive-scenarios3-how-would-you-handle-partially-observable-environments4-what-safety-considerations-are-important-in-rl-applicationspart-6-summary-and-key-takeaways-what-weve-learned1-markov-decision-processes-mdps--framework-sequential-decision-making-under-uncertainty--components-s-a-p-r-γ---states-actions-transitions-rewards-discount--markov-property-future-depends-only-on-current-state-not-history--foundation-mathematical-basis-for-all-rl-algorithms2-value-functions--vπs-expected-return-starting-from-state-s-following-policy-π---qπsa-expected-return-taking-action-a-in-state-s-then-following-π--relationship-vπs--σa-πas-qπsa--purpose-measure-goodness-of-states-and-actions3-bellman-equations--for-v-vπs--σa-πas-σs-pssarsas--γvπs--for-q-qπsa--σs-pssarsas--γ-σa-πasqπsa--significance-recursive-relationship-enabling-dynamic-programming-solutions4-policy-evaluation--algorithm-iterative-method-to-compute-vπ-given-policy-π--convergence-guaranteed-for-finite-mdps-with-γ--1--application-foundation-for-policy-iteration-and-value-iteration5-policy-improvement--theorem-greedy-policy-wrt-vπ-is-at-least-as-good-as-π--formula-πs--argmaxa-qπsa--monotonicity-each-improvement-step-yields-better-or-equal-policy6-policy-iteration--algorithm-alternates-between-evaluation-and-improvement--guarantee-converges-to-optimal-policy-π--efficiency-usually-converges-in-few-iterations----key-insights-from-experimentsdiscount-factor-γ-effects--low-γ-myopic-behavior-focuses-on-immediate-rewards--high-γ-farsighted-behavior-considers-long-term-consequences--trade-off-convergence-speed-vs-solution-qualityenvironment-structure-impact--reward-structure-significantly-affects-optimal-policy--obstacles-create-navigation-challenges-requiring-planning--starting-position-can-influence-learning-dynamicsalgorithm-characteristics--model-based-requires-knowledge-of-pssa-and-rsas--exact-solution-finds-truly-optimal-policy-unlike-approximate-methods--computational-cost-scales-with-state-space-size----connections-to-advanced-topicswhat-this-enables--value-iteration-direct-optimization-of-value-function--q-learning-model-free-learning-of-action-value-functions--deep-rl-neural-network-function-approximation--policy-gradients-direct-policy-optimization-methodsnext-steps-in-learning--temporal-difference-learning-learn-from-incomplete-episodes--function-approximation-handle-largecontinuous-state-spaces--exploration-vs-exploitation-balance-learning-and-performance--multi-agent-systems-multiple-learning-agents-interacting----reflection-questionstheoretical-understanding1-how-would-stochastic-transitions-affect-the-optimal-policy2-what-happens-with-continuous-state-or-action-spaces3-how-do-we-handle-unknown-environment-dynamics4-what-are-computational-limits-for-large-state-spacespractical-applications1-how-could-you-apply-mdps-to-real-world-decision-problems2-what-modifications-would-be-needed-for-competitive-scenarios3-how-would-you-handle-partially-observable-environments4-what-safety-considerations-are-important-in-rl-applications---common-misconceptions-and-intuitive-understandingbefore-we-wrap-up-lets-address-some-common-confusions-and-solidify-understanding--common-misconceptions1-value-functions-are-just-rewards---wrong-vs--rs----correct-vs--expected-total-future-reward-from-state-s---think-vs-is-like-your-bank-account-balance-rs-is-your-daily-income2-qsa-tells-me-the-best-action---wrong-qsa-is-not-binary-goodbad---correct-qsa-is-the-expected-value-of-taking-action-a---think-compare-q-values-to-choose-best-action-argmax_a-qsa3-policy-iteration-always-takes-many-steps---wrong-often-converges-in-2-4-iterations---correct-convergence-is-usually-very-fast---think-once-you-find-a-good-strategy-small-improvements-are-enough4-random-policy-is-always-bad---wrong-random-policy-can-be-good-for-exploration---correct-depends-on-environment-and-goals---think-sometimes-trying-new-things-leads-to-better-discoveries--key-intuitions-to-remember1-the-big-picture-flowenvironment--policy--actions--rewards--better-policy--repeat2-value-functions-as-gps--vs-how-good-is-this-location-overall--qsa-how-good-is-taking-this-road-from-this-location3-bellman-equations-as-consistency--my-value-should-equal-immediate-reward--discounted-future-value--like-my-wealth--todays-income--tomorrows-wealth4-policy-improvement-as-learning--if-i-know-what-each-action-leads-to-i-can-choose-better-actions--like-if-i-know-exam-results-for-each-study-method-i-can-study-better--troubleshooting-guideif-values-dont-converge--check-if-γ--1---reduce-convergence-threshold-theta--check-for-bugs-in-transition-probabilitiesif-policy-doesnt-improve--environment-might-be-too-simple-already-optimal--check-reward-structure---might-need-more-differentiation--verify-policy-improvement-logicif-results-seem-weird--visualize-value-functions-and-policies--start-with-simpler-environment--check-reward-signs-positivenegative--connecting-to-future-topicswhat-we-learned-here-enables--value-iteration-direct-value-optimization-next-week--q-learning-learn-q-values-without-knowing-the-model--deep-rl-use-neural-networks-to-handle-large-state-spaces--policy-gradients-directly-optimize-the-policy-parameters--the-rl-mindsetthink-like-an-rl-agent1-observe-your-current-situation-state2-consider-your-options-actions-3-predict-outcomes-use-your-modelexperience4-choose-the-best-option-policy5-learn-from-results-update-valuespolicy6-repeat-until-masterythis-mindset-applies-to--career-decisions--investment-choices---game-strategies--daily-life-optimization-common-misconceptions-and-intuitive-understandingbefore-we-wrap-up-lets-address-some-common-confusions-and-solidify-understanding--common-misconceptions1-value-functions-are-just-rewards---wrong-vs--rs----correct-vs--expected-total-future-reward-from-state-s---think-vs-is-like-your-bank-account-balance-rs-is-your-daily-income2-qsa-tells-me-the-best-action---wrong-qsa-is-not-binary-goodbad---correct-qsa-is-the-expected-value-of-taking-action-a---think-compare-q-values-to-choose-best-action-argmax_a-qsa3-policy-iteration-always-takes-many-steps---wrong-often-converges-in-2-4-iterations---correct-convergence-is-usually-very-fast---think-once-you-find-a-good-strategy-small-improvements-are-enough4-random-policy-is-always-bad---wrong-random-policy-can-be-good-for-exploration---correct-depends-on-environment-and-goals---think-sometimes-trying-new-things-leads-to-better-discoveries--key-intuitions-to-remember1-the-big-picture-flowenvironment--policy--actions--rewards--better-policy--repeat2-value-functions-as-gps--vs-how-good-is-this-location-overall--qsa-how-good-is-taking-this-road-from-this-location3-bellman-equations-as-consistency--my-value-should-equal-immediate-reward--discounted-future-value--like-my-wealth--todays-income--tomorrows-wealth4-policy-improvement-as-learning--if-i-know-what-each-action-leads-to-i-can-choose-better-actions--like-if-i-know-exam-results-for-each-study-method-i-can-study-better--troubleshooting-guideif-values-dont-converge--check-if-γ--1---reduce-convergence-threshold-theta--check-for-bugs-in-transition-probabilitiesif-policy-doesnt-improve--environment-might-be-too-simple-already-optimal--check-reward-structure---might-need-more-differentiation--verify-policy-improvement-logicif-results-seem-weird--visualize-value-functions-and-policies--start-with-simpler-environment--check-reward-signs-positivenegative--connecting-to-future-topicswhat-we-learned-here-enables--value-iteration-direct-value-optimization-next-week--q-learning-learn-q-values-without-knowing-the-model--deep-rl-use-neural-networks-to-handle-large-state-spaces--policy-gradients-directly-optimize-the-policy-parameters--the-rl-mindsetthink-like-an-rl-agent1-observe-your-current-situation-state2-consider-your-options-actions-3-predict-outcomes-use-your-modelexperience4-choose-the-best-option-policy5-learn-from-results-update-valuespolicy6-repeat-until-masterythis-mindset-applies-to--career-decisions--investment-choices---game-strategies--daily-life-optimization)
  - [Part 1: Theoretical Foundation### 1.1 Reinforcement Learning Framework**definition:**reinforcement Learning Is a Computational Approach to Learning from Interaction. the Key Elements Are:- **agent**: the Learner and Decision Maker - the Entity That Makes Choices- **environment**: the World the Agent Interacts with - Everything outside the Agent- **state (s)**: Current Situation of the Agent - Describes the Current Circumstances- **action (a)**: Choices Available to the Agent - Decisions That Can Be Made- **reward (r)**: Numerical Feedback from Environment - Immediate Feedback Signal- **policy (π)**: Agent's Strategy for Choosing Actions - Mapping from States to Actions**real-world Analogy:**think of Rl like Learning to Drive:- **agent** = the Driver (you)- **environment** = Roads, Traffic, Weather Conditions- **state** = Current Speed, Position, Traffic around You- **actions** = Accelerate, Brake, Turn Left/right- **reward** = Positive for Safe Driving, Negative for Accidents- **policy** = Your Driving Strategy (cautious, Aggressive, Etc.)---### 1.2 Markov Decision Process (mdp)**definition:**an Mdp Is Defined by the Tuple (S, A, P, R, Γ) Where:- **s**: Set of States - All Possible Situations the Agent Can Encounter- **a**: Set of Actions - All Possible Decisions Available to the Agent- **p**: Transition Probability Function P(s'|s,a) - Probability of Moving to State S' Given Current State S and Action A- **r**: Reward Function R(s,a,s') - Immediate Reward Received for Transitioning from S to S' Via Action A- **γ**: Discount Factor (0 ≤ Γ ≤ 1) - Determines Importance of Future Rewards**markov Property:**the Future Depends Only on the Current State, Not on the History of How We Got There. MATHEMATICALLY:P(S*{T+1} = S' | S*t = S, A*t = A, S*{T-1}, A*{T-1}, ..., S*0, A*0) = P(S*{T+1} = S' | S*t = S, A*t = A)**intuition:**the Current State Contains All Information Needed to Make Optimal Decisions. the past Is Already "encoded" in the Current State.---### 1.3 Value Functions**state-value Function:**$$v^π(s) = \mathbb{e}*π[g*t | S_t = S]$$**interpretation:** Expected Total Reward When Starting from State S and Following Policy Π. It Answers: "HOW Good Is It to Be in This State?"**action-value Function:**$$q^π(s,a) = \mathbb{e}*π[g*t | S*t = S, A*t = A]$$**interpretation:** Expected Total Reward When Taking Action a in State S and Then Following Policy Π. It Answers: "HOW Good Is It to Take This Specific Action in This State?"**return (cumulative Reward):**$$g*t = R*{T+1} + ΓR*{T+2} + Γ^2R*{T+3} + ... = \SUM*{K=0}^{\INFTY} Γ^k R*{T+K+1}$$**WHY Discount Factor Γ?**- **Γ = 0**: Only Immediate Rewards Matter (myopic)- **Γ = 1**: All Future Rewards Are Equally Important- **0 < Γ < 1**: Future Rewards Are Discounted (realistic for Most Scenarios)---### 1.4 Bellman Equations**bellman Equation for State-value Function:**$$v^π(s) = \sum*a Π(a|s) \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**interpretation:** the Value of a State Equals the Immediate Reward Plus the Discounted Value of the Next State, Averaged over All Possible Actions and Transitions.**bellman Equation for Action-value Function:**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γ \sum*{a'} Π(a'|s')q^π(s',a')]$$**key Insight:** the Bellman Equations Express a Recursive Relationship - the Value of a State (OR State-action Pair) Depends on the Immediate Reward Plus the Discounted Value of Future States. This Is the Mathematical Foundation for Most Rl Algorithms.](#part-1-theoretical-foundation-11-reinforcement-learning-frameworkdefinitionreinforcement-learning-is-a-computational-approach-to-learning-from-interaction-the-key-elements-are--agent-the-learner-and-decision-maker---the-entity-that-makes-choices--environment-the-world-the-agent-interacts-with---everything-outside-the-agent--state-s-current-situation-of-the-agent---describes-the-current-circumstances--action-a-choices-available-to-the-agent---decisions-that-can-be-made--reward-r-numerical-feedback-from-environment---immediate-feedback-signal--policy-π-agents-strategy-for-choosing-actions---mapping-from-states-to-actionsreal-world-analogythink-of-rl-like-learning-to-drive--agent--the-driver-you--environment--roads-traffic-weather-conditions--state--current-speed-position-traffic-around-you--actions--accelerate-brake-turn-leftright--reward--positive-for-safe-driving-negative-for-accidents--policy--your-driving-strategy-cautious-aggressive-etc----12-markov-decision-process-mdpdefinitionan-mdp-is-defined-by-the-tuple-s-a-p-r-γ-where--s-set-of-states---all-possible-situations-the-agent-can-encounter--a-set-of-actions---all-possible-decisions-available-to-the-agent--p-transition-probability-function-pssa---probability-of-moving-to-state-s-given-current-state-s-and-action-a--r-reward-function-rsas---immediate-reward-received-for-transitioning-from-s-to-s-via-action-a--γ-discount-factor-0--γ--1---determines-importance-of-future-rewardsmarkov-propertythe-future-depends-only-on-the-current-state-not-on-the-history-of-how-we-got-there-mathematicallypst1--s--st--s-at--a-st-1-at-1--s0-a0--pst1--s--st--s-at--aintuitionthe-current-state-contains-all-information-needed-to-make-optimal-decisions-the-past-is-already-encoded-in-the-current-state----13-value-functionsstate-value-functionvπs--mathbbeπgt--s_t--sinterpretation-expected-total-reward-when-starting-from-state-s-and-following-policy-π-it-answers-how-good-is-it-to-be-in-this-stateaction-value-functionqπsa--mathbbeπgt--st--s-at--ainterpretation-expected-total-reward-when-taking-action-a-in-state-s-and-then-following-policy-π-it-answers-how-good-is-it-to-take-this-specific-action-in-this-statereturn-cumulative-rewardgt--rt1--γrt2--γ2rt3----sumk0infty-γk-rtk1why-discount-factor-γ--γ--0-only-immediate-rewards-matter-myopic--γ--1-all-future-rewards-are-equally-important--0--γ--1-future-rewards-are-discounted-realistic-for-most-scenarios----14-bellman-equationsbellman-equation-for-state-value-functionvπs--suma-πas-sums-pssarsas--γvπsinterpretation-the-value-of-a-state-equals-the-immediate-reward-plus-the-discounted-value-of-the-next-state-averaged-over-all-possible-actions-and-transitionsbellman-equation-for-action-value-functionqπsa--sums-pssarsas--γ-suma-πasqπsakey-insight-the-bellman-equations-express-a-recursive-relationship---the-value-of-a-state-or-state-action-pair-depends-on-the-immediate-reward-plus-the-discounted-value-of-future-states-this-is-the-mathematical-foundation-for-most-rl-algorithms)
    - [📚 Common Misconceptions and Clarifications**misconception 1: "THE Agent Knows the Environment Model"**reality:** in Most Rl Problems, the Agent Doesn't Know P(s'|s,a) or R(s,a,s'). This Is Called "model-free" Rl, Where the Agent Learns through Trial and Error.**misconception 2: "higher Rewards Are Always Better"**reality:** the Goal Is to Maximize *cumulative* Reward, Not Immediate Reward. Sometimes Taking a Small Immediate Reward Prevents Getting a Much Larger Future Reward.**misconception 3: "THE Policy Should Always Be Deterministic"****reality:** Stochastic Policies (that Output Probabilities) Are Often Better Because They Allow for Exploration and Can Be Optimal in Certain Environments.---### 🧠 Building Intuition: Restaurant Example**scenario:** You're Choosing Restaurants to Visit in a New City.**mdp Components:**- **states**: Your Hunger Level, Location, Time of Day, Budget- **actions**: Choose Restaurant A, B, C, or Cook at Home- **rewards**: Satisfaction from Food (immediate) + Health Effects (long-term)- **transitions**: How Your State Changes after Eating**value Functions:**- **v(hungry, Downtown, Evening)**: How Good Is This Situation Overall?- **q(hungry, Downtown, Evening, "restaurant A")**: How Good Is Choosing Restaurant a in This Situation?**policy Learning:** Initially Random Choices → Gradually Prefer Restaurants That Gave Good Experiences → Eventually Develop a Strategy That Considers Health, Taste, Cost, and Convenience.---### 🔧 Mathematical Properties and Theorems**theorem 1: Existence and Uniqueness of Value Functions**for Any Policy Π and Finite Mdp, There Exists a Unique Solution to the Bellman Equations.**theorem 2: Bellman Optimality Principle**a Policy Π Is Optimal If and Only If:$$v^π(s) = \max_a Q^π(s,a) \text{ for All } S \IN S$$**theorem 3: Policy Improvement Theorem**if Π' Is Greedy with Respect to V^π, Then V^π'(s) ≥ V^π(s) for All States S.**practical Implications:**- We Can Always Improve a Policy by Being Greedy with Respect to Its Value Function- There Always Exists an Optimal Policy (MAY Not Be Unique)- the Optimal Value Function Satisfies the Bellman Optimality Equations- **rewards**: +1 for Safe Driving, -10 for Accidents, -1 for Speeding Tickets- **policy**: Your Driving Strategy (aggressive, Conservative, Etc.)#### **why Markov Property Matters**the **markov Property** Means "THE Future Depends Only on the Present, Not the Past."**example**: in Chess, to Decide Your Next Move, You Only Need to See the Current Board Position. You Don't Need to Know How the Pieces Got There - the Complete Game History Is Irrelevant for Making the Optimal Next Move.**non-markov Example**: Predicting Tomorrow's Weather Based Only on Today's Weather (YOU Need Historical Patterns).#### **understanding the Discount Factor (γ)**the Discount Factor Determines How Much You Care About Future Rewards:- **Γ = 0**: "I Only Care About Immediate Rewards" (very Myopic)- Example: Only Caring About This Month's Salary, Not Career Growth - **Γ = 0.9**: "future Rewards Are Worth 90% of Immediate Rewards"- Example: Investing Money - You Value Future Returns but Prefer Sooner - **Γ = 1**: "future Rewards Are as Valuable as Immediate Rewards"- Example: Climate Change Actions - Long-term Benefits Matter Equally**mathematical Impact**:- Return G*t = R*{T+1} + ΓR*{T+2} + Γ²R*{T+3} + ...- with Γ=0.9: G*t = R*{T+1} + 0.9×R*{T+2} + 0.81×R*{T+3} + ...- Future Rewards Get Progressively Less Important](#-common-misconceptions-and-clarificationsmisconception-1-the-agent-knows-the-environment-modelreality-in-most-rl-problems-the-agent-doesnt-know-pssa-or-rsas-this-is-called-model-free-rl-where-the-agent-learns-through-trial-and-errormisconception-2-higher-rewards-are-always-betterreality-the-goal-is-to-maximize-cumulative-reward-not-immediate-reward-sometimes-taking-a-small-immediate-reward-prevents-getting-a-much-larger-future-rewardmisconception-3-the-policy-should-always-be-deterministicreality-stochastic-policies-that-output-probabilities-are-often-better-because-they-allow-for-exploration-and-can-be-optimal-in-certain-environments-----building-intuition-restaurant-examplescenario-youre-choosing-restaurants-to-visit-in-a-new-citymdp-components--states-your-hunger-level-location-time-of-day-budget--actions-choose-restaurant-a-b-c-or-cook-at-home--rewards-satisfaction-from-food-immediate--health-effects-long-term--transitions-how-your-state-changes-after-eatingvalue-functions--vhungry-downtown-evening-how-good-is-this-situation-overall--qhungry-downtown-evening-restaurant-a-how-good-is-choosing-restaurant-a-in-this-situationpolicy-learning-initially-random-choices--gradually-prefer-restaurants-that-gave-good-experiences--eventually-develop-a-strategy-that-considers-health-taste-cost-and-convenience-----mathematical-properties-and-theoremstheorem-1-existence-and-uniqueness-of-value-functionsfor-any-policy-π-and-finite-mdp-there-exists-a-unique-solution-to-the-bellman-equationstheorem-2-bellman-optimality-principlea-policy-π-is-optimal-if-and-only-ifvπs--max_a-qπsa-text-for-all--s-in-stheorem-3-policy-improvement-theoremif-π-is-greedy-with-respect-to-vπ-then-vπs--vπs-for-all-states-spractical-implications--we-can-always-improve-a-policy-by-being-greedy-with-respect-to-its-value-function--there-always-exists-an-optimal-policy-may-not-be-unique--the-optimal-value-function-satisfies-the-bellman-optimality-equations--rewards-1-for-safe-driving--10-for-accidents--1-for-speeding-tickets--policy-your-driving-strategy-aggressive-conservative-etc-why-markov-property-mattersthe-markov-property-means-the-future-depends-only-on-the-present-not-the-pastexample-in-chess-to-decide-your-next-move-you-only-need-to-see-the-current-board-position-you-dont-need-to-know-how-the-pieces-got-there---the-complete-game-history-is-irrelevant-for-making-the-optimal-next-movenon-markov-example-predicting-tomorrows-weather-based-only-on-todays-weather-you-need-historical-patterns-understanding-the-discount-factor-γthe-discount-factor-determines-how-much-you-care-about-future-rewards--γ--0-i-only-care-about-immediate-rewards-very-myopic--example-only-caring-about-this-months-salary-not-career-growth---γ--09-future-rewards-are-worth-90-of-immediate-rewards--example-investing-money---you-value-future-returns-but-prefer-sooner---γ--1-future-rewards-are-as-valuable-as-immediate-rewards--example-climate-change-actions---long-term-benefits-matter-equallymathematical-impact--return-gt--rt1--γrt2--γ²rt3----with-γ09-gt--rt1--09rt2--081rt3----future-rewards-get-progressively-less-important)
    - [🎮 Understanding Our Gridworld Environmentbefore We Dive into the Code, Let's Understand What We're Building:#### **the Gridworld SETUP**```(0,0) → → → (0,3) ↓ X X ↓ ↓ X ◯ ↓ (3,0) → → → (3,3) 🎯```**legend:**- `S` at (0,0): Starting Position- `🎯` at (3,3): Goal (treasure!)- `X`: Obstacles (walls or Pits)- `◯`: Regular Empty Spaces- Arrows: Possible Movements#### **why This Environment Is Perfect for LEARNING**1. **small & Manageable**: 4×4 Grid = 16 States (easy to VISUALIZE)2. **clear Objective**: Get from Start to GOAL3. **interesting Obstacles**: Forces Strategic THINKING4. **deterministic**: Same Action Always Leads to Same Result (FOR Now)#### **reward Structure Explained**- **goal Reward (+10)**: Big Positive Reward for Reaching the Treasure- **step Penalty (-0.1)**: Small Negative Reward for Each Move (encourages Efficiency)- **obstacle Penalty (-5)**: Big Negative Reward for Hitting Obstacles (safety First!)**why These Specific Values?**- Goal Reward Is Much Larger Than Step Penalty → Encourages Reaching the Goal- Obstacle Penalty Is Significant → Discourages Dangerous Moves- Step Penalty Is Small → Prevents Infinite Wandering without Being Too Harsh#### **state Representation**each State Is a Tuple (row, Column):- (0,0) = Top-left Corner- (3,3) = Bottom-right Corner - States Are like Gps Coordinates for Our Agent](#-understanding-our-gridworld-environmentbefore-we-dive-into-the-code-lets-understand-what-were-building-the-gridworld-setup00----03--x-x---x---30----33-legend--s-at-00-starting-position---at-33-goal-treasure--x-obstacles-walls-or-pits---regular-empty-spaces--arrows-possible-movements-why-this-environment-is-perfect-for-learning1-small--manageable-44-grid--16-states-easy-to-visualize2-clear-objective-get-from-start-to-goal3-interesting-obstacles-forces-strategic-thinking4-deterministic-same-action-always-leads-to-same-result-for-now-reward-structure-explained--goal-reward-10-big-positive-reward-for-reaching-the-treasure--step-penalty--01-small-negative-reward-for-each-move-encourages-efficiency--obstacle-penalty--5-big-negative-reward-for-hitting-obstacles-safety-firstwhy-these-specific-values--goal-reward-is-much-larger-than-step-penalty--encourages-reaching-the-goal--obstacle-penalty-is-significant--discourages-dangerous-moves--step-penalty-is-small--prevents-infinite-wandering-without-being-too-harsh-state-representationeach-state-is-a-tuple-row-column--00--top-left-corner--33--bottom-right-corner---states-are-like-gps-coordinates-for-our-agent)
  - [Part 2: Policy Definition and Evaluation### Exercise 2.1: Define Different Policies**definition:**a Policy Π(a|s) Defines the Probability of Taking Action a in State S. It's the Agent's Strategy for Choosing Actions.**mathematical Representation:**$$\pi(a|s) = P(\text{action} = a | \text{state} = S)$$**types of Policies:**- **deterministic Policy**: Π(a|s) ∈ {0, 1} - Always Chooses the Same Action in a Given State- **stochastic Policy**: Π(a|s) ∈ [0, 1] - Chooses Actions Probabilistically**policies We'll IMPLEMENT:**1. **random Policy**: Equal Probability for All Valid ACTIONS2. **greedy Policy**: Always Move towards the Goal 3. **custom Policy**: Your Own Strategic Policy---### Exercise 2.2: Policy Evaluation**definition:**policy Evaluation Computes the Value Function V^π(s) for a Given Policy Π. It Answers: "HOW Good Is This Policy?"**iterative Policy Evaluation ALGORITHM:**1. **initialize**: V(s) = 0 for All States S2. **repeat until Convergence**:- for Each State S:- V*new(s) = Σ*a Π(a|s) Σ*{s'} P(s'|s,a)[r(s,a,s') + ΓV(S')]3. **return**: Converged Value Function V**convergence Condition:**max*s |v*new(s) - V*old(s)| < Θ (where Θ Is a Small Threshold, E.g., 1E-6)**INTUITION:**WE Start with All State Values at Zero and Iteratively Update Them Based on the Bellman Equation until They Stabilize. It's like Repeatedly Asking "IF I Follow This Policy, How Much Reward Will I Get?" until the Answer Stops Changing.](#part-2-policy-definition-and-evaluation-exercise-21-define-different-policiesdefinitiona-policy-πas-defines-the-probability-of-taking-action-a-in-state-s-its-the-agents-strategy-for-choosing-actionsmathematical-representationpias--ptextaction--a--textstate--stypes-of-policies--deterministic-policy-πas--0-1---always-chooses-the-same-action-in-a-given-state--stochastic-policy-πas--0-1---chooses-actions-probabilisticallypolicies-well-implement1-random-policy-equal-probability-for-all-valid-actions2-greedy-policy-always-move-towards-the-goal-3-custom-policy-your-own-strategic-policy----exercise-22-policy-evaluationdefinitionpolicy-evaluation-computes-the-value-function-vπs-for-a-given-policy-π-it-answers-how-good-is-this-policyiterative-policy-evaluation-algorithm1-initialize-vs--0-for-all-states-s2-repeat-until-convergence--for-each-state-s--vnews--σa-πas-σs-pssarsas--γvs3-return-converged-value-function-vconvergence-conditionmaxs-vnews---volds--θ-where-θ-is-a-small-threshold-eg-1e-6intuitionwe-start-with-all-state-values-at-zero-and-iteratively-update-them-based-on-the-bellman-equation-until-they-stabilize-its-like-repeatedly-asking-if-i-follow-this-policy-how-much-reward-will-i-get-until-the-answer-stops-changing)
    - [🧭 Policy Deep Dive: Understanding Different Strategies**what Is a Policy?**a Policy Is like a Gps Navigation System for Our Agent. It Tells the Agent What to Do in Every Possible Situation.**mathematical Definition:**π(a|s) = Probability of Taking Action a When in State S---### 📋 Types of Policies We'll IMPLEMENT**1. Random Policy** 🎲**strategy:** "when in Doubt, Flip a Coin"**mathematical Definition:** Π(a|s) = 1/|VALID_ACTIONS| for All Valid Actions**example:** at State (1,0), If We Can Go [UP, Down, Right], Each Has 33.33% Probability**advantages:**- Explores All Possibilities Equally- Simple to Implement- Guarantees Exploration**disadvantages:**- Not Very Efficient- like Wandering Randomly in a Maze- No Learning from EXPERIENCE---**2. Greedy Policy** 🎯**strategy:** "always Move Closer to the Goal"**mathematical Definition:** Π(a|s) = 1 If a Minimizes Distance to Goal, 0 Otherwise**example:** at State (1,0), If Goal Is at (3,3), Prefer "down" and "right"**advantages:**- Very Efficient When It Works- Direct Path to Goal- Fast Convergence**disadvantages:**- Can Get Stuck in Local Optima- Might Walk into Obstacles- No Exploration of Alternative PATHS---**3. Custom Policy** 🎨**strategy:** Your Creative Combination of Strategies**examples:**- **epsilon-greedy**: 90% Greedy, 10% Random- **safety-first**: Avoid Actions That Lead near Obstacles- **wall-follower**: Stay Close to Boundaries---### 🎮 Real-world Analogies**policy Vs Strategy in Games:**think of Different Video Game Playing Styles:- **aggressive Player**: Always Attacks (deterministic Policy)- **defensive Player**: Always Defends (deterministic Policy)- **adaptive Player**: 70% Attack, 30% Defend (stochastic Policy)**why Stochastic Policies?**sometimes Randomness Helps:- **exploration**: Discover New Paths You Wouldn't Normally Try- **unpredictability**: in Competitive Games, Being Predictable Is Bad- **robustness**: Handle Uncertainty in the Environment**restaurant Choice Analogy:**- **random Policy**: Pick Restaurants Randomly- **greedy Policy**: Always Go to Your Current Favorite- **epsilon-greedy Policy**: Usually Go to Favorite, Sometimes Try Something New](#-policy-deep-dive-understanding-different-strategieswhat-is-a-policya-policy-is-like-a-gps-navigation-system-for-our-agent-it-tells-the-agent-what-to-do-in-every-possible-situationmathematical-definitionπas--probability-of-taking-action-a-when-in-state-s-----types-of-policies-well-implement1-random-policy-strategy-when-in-doubt-flip-a-coinmathematical-definition-πas--1valid_actions-for-all-valid-actionsexample-at-state-10-if-we-can-go-up-down-right-each-has-3333-probabilityadvantages--explores-all-possibilities-equally--simple-to-implement--guarantees-explorationdisadvantages--not-very-efficient--like-wandering-randomly-in-a-maze--no-learning-from-experience---2-greedy-policy-strategy-always-move-closer-to-the-goalmathematical-definition-πas--1-if-a-minimizes-distance-to-goal-0-otherwiseexample-at-state-10-if-goal-is-at-33-prefer-down-and-rightadvantages--very-efficient-when-it-works--direct-path-to-goal--fast-convergencedisadvantages--can-get-stuck-in-local-optima--might-walk-into-obstacles--no-exploration-of-alternative-paths---3-custom-policy-strategy-your-creative-combination-of-strategiesexamples--epsilon-greedy-90-greedy-10-random--safety-first-avoid-actions-that-lead-near-obstacles--wall-follower-stay-close-to-boundaries-----real-world-analogiespolicy-vs-strategy-in-gamesthink-of-different-video-game-playing-styles--aggressive-player-always-attacks-deterministic-policy--defensive-player-always-defends-deterministic-policy--adaptive-player-70-attack-30-defend-stochastic-policywhy-stochastic-policiessometimes-randomness-helps--exploration-discover-new-paths-you-wouldnt-normally-try--unpredictability-in-competitive-games-being-predictable-is-bad--robustness-handle-uncertainty-in-the-environmentrestaurant-choice-analogy--random-policy-pick-restaurants-randomly--greedy-policy-always-go-to-your-current-favorite--epsilon-greedy-policy-usually-go-to-favorite-sometimes-try-something-new)
    - [🔍 Understanding Policy Evaluation Step-by-steppolicy Evaluation Answers the Question: **"how Good Is Each State If I Follow This Policy?"**#### **the Intuition**imagine You're Evaluating Different Starting Positions in a Board Game:- Some Positions Are Naturally Better (closer to Winning)- Some Positions Are Worse (closer to Losing) - the "value" of a Position Depends on How Well You'll Do from There#### **mathematical Breakdown****the Bellman Equation for State Values:**```v^π(s) = Σ*a Π(a|s) × Σ*{s'} P(s'|s,a) × [r(s,a,s') + Γ × V^π(s')]```**let's Decode This Step by STEP:**1. **for Each Possible Action A**: Π(a|s) = "HOW Likely Am I to Take Action a in State S?"2. **for Each Possible Next State S'**: P(s'|s,a) = "IF I Take Action A, What's the Chance I End Up in State S'?"3. **calculate Immediate Reward + Future Value**: R(s,a,s') + Γ × V^π(s')- R(s,a,s') = "what Reward Do I Get Immediately?"- Γ × V^π(s') = "what's the Discounted Future VALUE?"4. **sum Everything Up**: This Gives the Expected Value of Being in State S#### **simple Example**let's Say We're at State (2,2) with a Random Policy:```python# Random Policy: Equal Probability for All Valid Actionsπ(up|s) = 0.25, Π(down|s) = 0.25, Π(left|s) = 0.25, Π(right|s) = 0.25# FOR Action "UP" → Next State (1,2)CONTRIBUTION*UP = 0.25 × 1.0 × (-0.1 + 0.9 × V(1,2))# FOR Action "down" → Next State (3,2)CONTRIBUTION*DOWN = 0.25 × 1.0 × (-0.1 + 0.9 × V(3,2))# ... and So on for Left and RIGHTV(2,2) = Contribution*up + Contribution*down + Contribution*left + Contribution*right```#### **why Iterative?**- We Start with V(s) = 0 for All States (initial Guess)- Each Iteration Improves Our Estimate Using Current Values- Eventually, Values Converge to True Values- like Asking "IF I Knew the Value of My Neighbors, What Would My Value Be?"#### **convergence Intuition**think of It like Gossip Spreading in a Neighborhood:- Initially, Nobody Knows the True "gossip" (values)- Each Iteration, Neighbors Share Information - Eventually, Everyone Converges to the Same True Story](#-understanding-policy-evaluation-step-by-steppolicy-evaluation-answers-the-question-how-good-is-each-state-if-i-follow-this-policy-the-intuitionimagine-youre-evaluating-different-starting-positions-in-a-board-game--some-positions-are-naturally-better-closer-to-winning--some-positions-are-worse-closer-to-losing---the-value-of-a-position-depends-on-how-well-youll-do-from-there-mathematical-breakdownthe-bellman-equation-for-state-valuesvπs--σa-πas--σs-pssa--rsas--γ--vπslets-decode-this-step-by-step1-for-each-possible-action-a-πas--how-likely-am-i-to-take-action-a-in-state-s2-for-each-possible-next-state-s-pssa--if-i-take-action-a-whats-the-chance-i-end-up-in-state-s3-calculate-immediate-reward--future-value-rsas--γ--vπs--rsas--what-reward-do-i-get-immediately--γ--vπs--whats-the-discounted-future-value4-sum-everything-up-this-gives-the-expected-value-of-being-in-state-s-simple-examplelets-say-were-at-state-22-with-a-random-policypython-random-policy-equal-probability-for-all-valid-actionsπups--025-πdowns--025-πlefts--025-πrights--025-for-action-up--next-state-12contributionup--025--10---01--09--v12-for-action-down--next-state-32contributiondown--025--10---01--09--v32--and-so-on-for-left-and-rightv22--contributionup--contributiondown--contributionleft--contributionright-why-iterative--we-start-with-vs--0-for-all-states-initial-guess--each-iteration-improves-our-estimate-using-current-values--eventually-values-converge-to-true-values--like-asking-if-i-knew-the-value-of-my-neighbors-what-would-my-value-be-convergence-intuitionthink-of-it-like-gossip-spreading-in-a-neighborhood--initially-nobody-knows-the-true-gossip-values--each-iteration-neighbors-share-information---eventually-everyone-converges-to-the-same-true-story)
    - [Exercise 2.3: Create Your Custom Policy**task**: Design and Implement Your Own Policy. Consider Strategies Like:- **wall-following**: Try to Stay Close to Walls- **risk-averse**: Avoid Obstacles with Higher Probability- **exploration-focused**: Balance between Moving towards Goal and Exploring**your Implementation Below**:](#exercise-23-create-your-custom-policytask-design-and-implement-your-own-policy-consider-strategies-like--wall-following-try-to-stay-close-to-walls--risk-averse-avoid-obstacles-with-higher-probability--exploration-focused-balance-between-moving-towards-goal-and-exploringyour-implementation-below)
  - [Part 3: Action-value Functions (q-functions)### Exercise 3.1: Computing Q-values**definition:**the Action-value Function Q^π(s,a) Represents the Expected Return When Taking Action a in State S and Then Following Policy Π.**key Question Q-functions Answer:**q-functions Answer: "what If I Take This Specific Action Here, Then Follow My Policy?"**mathematical Relationships:****v from Q (policy-weighted Average):**$$v^π(s) = \sum*a Π(a|s) Q^π(s,a)$$**q from V (bellman Backup):**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**bellman Equation for Q:**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γ \sum*{a'} Π(a'|s')q^π(s',a')]$$**intuition:**- **v(s)**: "HOW Good Is This State?" (following Current Policy)- **q(s,a)**: "HOW Good Is This Specific Action?" (then Following Policy)the V-q Relationship Is like Asking:- V: "HOW Well Will I Do from This Chess Position?"- Q: "HOW Well Will I Do If I Move My Queen Here, Then Play Normally?"](#part-3-action-value-functions-q-functions-exercise-31-computing-q-valuesdefinitionthe-action-value-function-qπsa-represents-the-expected-return-when-taking-action-a-in-state-s-and-then-following-policy-πkey-question-q-functions-answerq-functions-answer-what-if-i-take-this-specific-action-here-then-follow-my-policymathematical-relationshipsv-from-q-policy-weighted-averagevπs--suma-πas-qπsaq-from-v-bellman-backupqπsa--sums-pssarsas--γvπsbellman-equation-for-qqπsa--sums-pssarsas--γ-suma-πasqπsaintuition--vs-how-good-is-this-state-following-current-policy--qsa-how-good-is-this-specific-action-then-following-policythe-v-q-relationship-is-like-asking--v-how-well-will-i-do-from-this-chess-position--q-how-well-will-i-do-if-i-move-my-queen-here-then-play-normally)
    - [🎯 Q-functions Deep Dive: the "what If" Values**core Concept:**q-functions Provide Action-specific Evaluations, Allowing Us to Compare Different Choices Directly.---### 🍕 Restaurant Decision Analogy**scenario:** You're Choosing a Restaurant from Downtown Location.**value Functions:**- **v(downtown)** = 7.5 → "average Satisfaction from This Location with My Usual Choices"- **q(downtown, Pizza*place)** = 8.2 → "satisfaction If I Specifically Choose Pizza"- **q(downtown, Sushi*place)** = 6.8 → "satisfaction If I Specifically Choose Sushi"- **q(downtown, Burger*place)** = 7.1 → "satisfaction If I Specifically Choose Burgers"**policy Calculation:**if My Policy Is 50% Pizza, 30% Sushi, 20% Burgers:v(downtown) = 0.5×8.2 + 0.3×6.8 + 0.2×7.1 = 4.1 + 2.04 + 1.42 = 7.56 ✓---### 🧮 Mathematical Relationships EXPLAINED**1. V from Q (weighted Average):**$$v^π(s) = \sum*a Π(a|s) × Q^π(s,a)$$**interpretation:** State Value = Probability of Each Action × Value of That ACTION**2. Q from V (bellman Backup):**$$q^π(s,a) = \sum*{s'} P(s'|s,a) × [r(s,a,s') + Γv^π(s')]$$**interpretation:** Action Value = Immediate Reward + Discounted Future State Value---### 🔥 Why Q-functions MATTER**1. Direct Action Comparison:**- Q(s, Left) = 5.2 Vs Q(s, Right) = 7.8 → Choose Right!- No Need to Compute State Values FIRST**2. Policy Improvement:**- Π*new(s) = Argmax*a Q^π*old(s,a)- Directly Find the Best ACTION**3. Optimal Decision Making:**- Q*(s,a) Tells Us the Value of Each Action under Optimal Behavior- Essential for Q-learning Algorithms---### 📊 Visual Understandingthink of Q-values as Action-specific "heat Maps":- **hot Spots** (high Q-values): Good Actions to Take- **cold Spots** (LOW Q-values): Actions to Avoid- **separate Map for Each Action**: Q(s,↑), Q(s,↓), Q(s,←), Q(s,→)**gridworld Example:**- Q(state, "toward*goal") Typically Has Higher Values- Q(state, "toward*obstacle") Typically Has Lower Values- Q(state, "toward*wall") Often Has Negative Values - Like: "restaurant Satisfaction = Meal Quality + How I'll Feel Tomorrow"#### **why Q-functions MATTER**1. **better Decision Making**: Q-values Directly Tell Us Which Action Is Best- Max*a Q(s,a) Gives the Best Action in State S2. **policy Improvement**: We Can Improve Policies by Being Greedy W.r.t. Q-values- Π*new(s) = Argmax*a Q^Π_OLD(S,A)3. **action Comparison**: Compare Different Actions in the Same State- "should I Go Left or Right from Here?"#### **visual Understanding**think of Q-values as a "heat Map" for Each Action:- **hot Spots** (high Q-values): Good Actions to Take- **cold Spots** (LOW Q-values): Actions to Avoid - **different Maps for Each Action**: Q(s,up), Q(s,down), Q(s,left), Q(s,right)#### **common Confusion: V Vs Q**- **v(s)**: "HOW Good Is My Current Strategy from This Position?"- **q(s,a)**: "HOW Good Is This Specific Move, Then Using My Strategy?"it's like Asking:- V: "HOW Well Will I Do in This Chess Position?" - Q: "HOW Well Will I Do If I Move My Queen Here, Then Play Normally?"](#-q-functions-deep-dive-the-what-if-valuescore-conceptq-functions-provide-action-specific-evaluations-allowing-us-to-compare-different-choices-directly-----restaurant-decision-analogyscenario-youre-choosing-a-restaurant-from-downtown-locationvalue-functions--vdowntown--75--average-satisfaction-from-this-location-with-my-usual-choices--qdowntown-pizzaplace--82--satisfaction-if-i-specifically-choose-pizza--qdowntown-sushiplace--68--satisfaction-if-i-specifically-choose-sushi--qdowntown-burgerplace--71--satisfaction-if-i-specifically-choose-burgerspolicy-calculationif-my-policy-is-50-pizza-30-sushi-20-burgersvdowntown--0582--0368--0271--41--204--142--756------mathematical-relationships-explained1-v-from-q-weighted-averagevπs--suma-πas--qπsainterpretation-state-value--probability-of-each-action--value-of-that-action2-q-from-v-bellman-backupqπsa--sums-pssa--rsas--γvπsinterpretation-action-value--immediate-reward--discounted-future-state-value-----why-q-functions-matter1-direct-action-comparison--qs-left--52-vs-qs-right--78--choose-right--no-need-to-compute-state-values-first2-policy-improvement--πnews--argmaxa-qπoldsa--directly-find-the-best-action3-optimal-decision-making--qsa-tells-us-the-value-of-each-action-under-optimal-behavior--essential-for-q-learning-algorithms-----visual-understandingthink-of-q-values-as-action-specific-heat-maps--hot-spots-high-q-values-good-actions-to-take--cold-spots-low-q-values-actions-to-avoid--separate-map-for-each-action-qs-qs-qs-qsgridworld-example--qstate-towardgoal-typically-has-higher-values--qstate-towardobstacle-typically-has-lower-values--qstate-towardwall-often-has-negative-values---like-restaurant-satisfaction--meal-quality--how-ill-feel-tomorrow-why-q-functions-matter1-better-decision-making-q-values-directly-tell-us-which-action-is-best--maxa-qsa-gives-the-best-action-in-state-s2-policy-improvement-we-can-improve-policies-by-being-greedy-wrt-q-values--πnews--argmaxa-qπ_oldsa3-action-comparison-compare-different-actions-in-the-same-state--should-i-go-left-or-right-from-here-visual-understandingthink-of-q-values-as-a-heat-map-for-each-action--hot-spots-high-q-values-good-actions-to-take--cold-spots-low-q-values-actions-to-avoid---different-maps-for-each-action-qsup-qsdown-qsleft-qsright-common-confusion-v-vs-q--vs-how-good-is-my-current-strategy-from-this-position--qsa-how-good-is-this-specific-move-then-using-my-strategyits-like-asking--v-how-well-will-i-do-in-this-chess-position---q-how-well-will-i-do-if-i-move-my-queen-here-then-play-normally)
  - [Part 4: Policy Improvement and Policy Iteration### Exercise 4.1: Policy Improvement**definition:**given a Value Function V^π, We Can Improve the Policy by Being Greedy with Respect to the Action-value Function.**policy Improvement Formula:**$$π'(s) = \arg\max*a Q^π(s,a) = \arg\max*a \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**interpretation:** Choose the Action That Maximizes Expected Return from Each State.**policy Improvement Theorem:**if Π' Is Greedy with Respect to V^π, Then V^π'(s) ≥ V^π(s) for All States S.**translation:** "IF I Always Choose the Best Action Based on My Current Understanding, I Can Only Do Better (OR at Least as Well)."---### Exercise 4.2: Policy Iteration Algorithm**policy Iteration STEPS:**1. **initialize**: Start with Arbitrary Policy Π₀2. **repeat until Convergence**:- **policy Evaluation**: Compute V^π*k (solve Bellman Equation)- **policy Improvement**: Π*{K+1}(S) = Argmax*a Q^Π_K(S,A)3. **output**: Optimal Policy Π* and Value Function V***convergence Guarantee:** Policy Iteration Is Guaranteed to Converge to the Optimal Policy in Finite Time for Finite Mdps.**why It Works:**- Each Step Produces a Better (OR Equal) Policy- There Are Only Finitely Many Deterministic Policies- Must Eventually Reach Optimal Policy](#part-4-policy-improvement-and-policy-iteration-exercise-41-policy-improvementdefinitiongiven-a-value-function-vπ-we-can-improve-the-policy-by-being-greedy-with-respect-to-the-action-value-functionpolicy-improvement-formulaπs--argmaxa-qπsa--argmaxa-sums-pssarsas--γvπsinterpretation-choose-the-action-that-maximizes-expected-return-from-each-statepolicy-improvement-theoremif-π-is-greedy-with-respect-to-vπ-then-vπs--vπs-for-all-states-stranslation-if-i-always-choose-the-best-action-based-on-my-current-understanding-i-can-only-do-better-or-at-least-as-well----exercise-42-policy-iteration-algorithmpolicy-iteration-steps1-initialize-start-with-arbitrary-policy-π₀2-repeat-until-convergence--policy-evaluation-compute-vπk-solve-bellman-equation--policy-improvement-πk1s--argmaxa-qπ_ksa3-output-optimal-policy-π-and-value-function-vconvergence-guarantee-policy-iteration-is-guaranteed-to-converge-to-the-optimal-policy-in-finite-time-for-finite-mdpswhy-it-works--each-step-produces-a-better-or-equal-policy--there-are-only-finitely-many-deterministic-policies--must-eventually-reach-optimal-policy)
    - [🚀 Policy Improvement Deep Dive: Making Better Decisions**core Idea:** Use the Value Function to Make Better Action Choices.---### 📚 Learning Process Analogy**scenario:** You're Learning to Play Chess.**policy Evaluation:** "HOW Good Is My Current Playing Style?"- Analyze Your Current Strategy- Evaluate Typical Game Outcomes- Identify Strengths and Weaknesses**policy Improvement:** "HOW Can I Play Better?"- Look at Each Position Where You Made Suboptimal Moves- Replace Bad Moves with Better Alternatives- Update Your Playing Strategy**policy Iteration:** Repeat This Cycle until You Can't Improve Further.---### 🧮 Mathematical Foundations**policy Improvement Theorem:**if Π' Is Greedy W.r.t. V^π, Then V^π'(s) ≥ V^π(s) for All S.**proof INTUITION:**1. **greedy Action**: Choose a Such That Q^π(s,a) Is MAXIMIZED2. **definition**: Q^π(s,a) ≥ V^π(s) for the Chosen ACTION3. **new Policy**: Π'(s) Gives This Optimal ACTION4. **result**: V^π'(s) ≥ V^π(s)**why Greedy Improvement Works:**- Current Policy Chooses Actions with Average Value V^π(s)- Greedy Policy Chooses Action with Maximum Value Q^π(s,a)- Maximum ≥ Average, So New Policy Is Better---### 🔄 Policy Iteration: the Complete Cycle**step 1 - Policy Evaluation:** "HOW Good Is My Current Policy?"```v^π(s) ← Expected Return Following Π from State S```**step 2 - Policy Improvement:** "what's the Best Action in Each State?"```π'(s) ← Action That Maximizes Q^π(s,a)```**step 3 - Check Convergence:** "DID My Policy Change?"```if Π'(s) = Π(s) for All S: Stop (optimal Found)else: Π ← Π' and Repeat```---### 🎯 Key PROPERTIES**1. Monotonic IMPROVEMENT:**V^Π₀ ≤ V^Π₁ ≤ V^Π₂ ≤ ... ≤ V^Π**2. Finite Convergence:**algorithm Terminates in Finite Steps (FOR Finite MDPS)**3. Optimal Solution:**final Policy Π* Is Optimal: V^π* = V**4. Model-based:**requires Knowledge of Transition Probabilities P(s'|s,a) and Rewards R(s,a,s')think of a Student Improving Their Study STRATEGY:1. **current Strategy** (policy Π): "I Study Randomly for 2 HOURS"2. **evaluate Strategy** (policy Evaluation): "HOW Well Does This Work for Each Subject?" 3. **find Better Strategy** (policy Improvement): "math Needs 3 Hours, History Needs 1 HOUR"4. **repeat**: Keep Refining until No More Improvements Possible#### **mathematical Intuition**policy Improvement Theorem**: If Q^π(s,a) > V^π(s) for Some Action A, Then Taking Action a Is Better Than Following Policy Π.**translation**: "IF Doing Action a Gives Higher Value Than My Current Average, I Should Do Action a More Often!"**greedy Improvement**:```pythonπ*new(s) = Argmax*a Q^π(s,a)```"always Choose the Action with Highest Q-value"#### **why Does This Work?**monotonic Improvement**: Each Policy Improvement Step Makes the Policy at Least as Good, Usually Better.**proof Sketch**:- If We're Greedy W.r.t. Q^π, We Get V^π_new ≥ V^π- "IF I Always Choose the Best Available Action, I Can't Do Worse"#### **policy Iteration: the Complete Algorithm**the Cycle**:```random Policy → Evaluate → Improve → Evaluate → Improve → ... → Optimal Policy```**why It CONVERGES**:1. **finite State/action Space**: Limited Number of Possible POLICIES2. **monotonic Improvement**: Each Step Makes Policy Better (OR SAME)3. **NO Cycles**: Can't Go Backwards to a Worse POLICY4. **must Terminate**: Eventually Reach Optimal Policy#### **real-world Example: Learning to Drive**iteration 1**:- **policy**: "drive Slowly Everywhere" - **evaluation**: "safe but Inefficient on Highways"- **improvement**: "drive Fast on Highways, Slow in Neighborhoods"**iteration 2**:- **policy**: "speed Varies by Road Type"- **evaluation**: "good, but Inefficient in Traffic" - **improvement**: "also Consider Traffic Conditions"**final Policy**: "optimal Speed Based on Road Type, Traffic, Weather, Etc."#### **key INSIGHTS**1. **guaranteed Improvement**: Policy Iteration Always Finds the Optimal Policy (FOR Finite MDPS)2. **fast Convergence**: Usually Converges in Just a Few ITERATIONS3. **NO Exploration Needed**: Uses Complete Model Knowledge (unlike Q-learning LATER)4. **computational Cost**: Each Iteration Requires Solving the Bellman Equation#### **common Pitfalls**- **getting Stuck**: in Stochastic Environments, Might Need Exploration- **computational Cost**: Policy Evaluation Can Be Expensive - **model Required**: Need to Know P(s'|s,a) and R(s,a,s')](#-policy-improvement-deep-dive-making-better-decisionscore-idea-use-the-value-function-to-make-better-action-choices-----learning-process-analogyscenario-youre-learning-to-play-chesspolicy-evaluation-how-good-is-my-current-playing-style--analyze-your-current-strategy--evaluate-typical-game-outcomes--identify-strengths-and-weaknessespolicy-improvement-how-can-i-play-better--look-at-each-position-where-you-made-suboptimal-moves--replace-bad-moves-with-better-alternatives--update-your-playing-strategypolicy-iteration-repeat-this-cycle-until-you-cant-improve-further-----mathematical-foundationspolicy-improvement-theoremif-π-is-greedy-wrt-vπ-then-vπs--vπs-for-all-sproof-intuition1-greedy-action-choose-a-such-that-qπsa-is-maximized2-definition-qπsa--vπs-for-the-chosen-action3-new-policy-πs-gives-this-optimal-action4-result-vπs--vπswhy-greedy-improvement-works--current-policy-chooses-actions-with-average-value-vπs--greedy-policy-chooses-action-with-maximum-value-qπsa--maximum--average-so-new-policy-is-better-----policy-iteration-the-complete-cyclestep-1---policy-evaluation-how-good-is-my-current-policyvπs--expected-return-following-π-from-state-sstep-2---policy-improvement-whats-the-best-action-in-each-stateπs--action-that-maximizes-qπsastep-3---check-convergence-did-my-policy-changeif-πs--πs-for-all-s-stop-optimal-foundelse-π--π-and-repeat-----key-properties1-monotonic-improvementvπ₀--vπ₁--vπ₂----vπ2-finite-convergencealgorithm-terminates-in-finite-steps-for-finite-mdps3-optimal-solutionfinal-policy-π-is-optimal-vπ--v4-model-basedrequires-knowledge-of-transition-probabilities-pssa-and-rewards-rsasthink-of-a-student-improving-their-study-strategy1-current-strategy-policy-π-i-study-randomly-for-2-hours2-evaluate-strategy-policy-evaluation-how-well-does-this-work-for-each-subject-3-find-better-strategy-policy-improvement-math-needs-3-hours-history-needs-1-hour4-repeat-keep-refining-until-no-more-improvements-possible-mathematical-intuitionpolicy-improvement-theorem-if-qπsa--vπs-for-some-action-a-then-taking-action-a-is-better-than-following-policy-πtranslation-if-doing-action-a-gives-higher-value-than-my-current-average-i-should-do-action-a-more-oftengreedy-improvementpythonπnews--argmaxa-qπsaalways-choose-the-action-with-highest-q-value-why-does-this-workmonotonic-improvement-each-policy-improvement-step-makes-the-policy-at-least-as-good-usually-betterproof-sketch--if-were-greedy-wrt-qπ-we-get-vπ_new--vπ--if-i-always-choose-the-best-available-action-i-cant-do-worse-policy-iteration-the-complete-algorithmthe-cyclerandom-policy--evaluate--improve--evaluate--improve----optimal-policywhy-it-converges1-finite-stateaction-space-limited-number-of-possible-policies2-monotonic-improvement-each-step-makes-policy-better-or-same3-no-cycles-cant-go-backwards-to-a-worse-policy4-must-terminate-eventually-reach-optimal-policy-real-world-example-learning-to-driveiteration-1--policy-drive-slowly-everywhere---evaluation-safe-but-inefficient-on-highways--improvement-drive-fast-on-highways-slow-in-neighborhoodsiteration-2--policy-speed-varies-by-road-type--evaluation-good-but-inefficient-in-traffic---improvement-also-consider-traffic-conditionsfinal-policy-optimal-speed-based-on-road-type-traffic-weather-etc-key-insights1-guaranteed-improvement-policy-iteration-always-finds-the-optimal-policy-for-finite-mdps2-fast-convergence-usually-converges-in-just-a-few-iterations3-no-exploration-needed-uses-complete-model-knowledge-unlike-q-learning-later4-computational-cost-each-iteration-requires-solving-the-bellman-equation-common-pitfalls--getting-stuck-in-stochastic-environments-might-need-exploration--computational-cost-policy-evaluation-can-be-expensive---model-required-need-to-know-pssa-and-rsas)
  - [Part 5: Experiments and Analysis### Exercise 5.1: Effect of Discount Factor (γ)**definition:**the Discount Factor Γ Determines How Much We Value Future Rewards Compared to Immediate Rewards.**mathematical Impact:**$$g*t = R*{T+1} + ΓR*{T+2} + Γ^2R*{T+3} + ... = \SUM*{K=0}^{\INFTY} Γ^k R*{T+K+1}$$**INTERPRETATION of Different Values:**- **Γ = 0**: Only Immediate Rewards Matter (myopic Behavior)- **Γ = 1**: All Future Rewards Equally Important (infinite Horizon)- **0 < Γ < 1**: Future Rewards Are Discounted (realistic)**task:** Experiment with Different Discount Factors and Analyze Their Effect on the Optimal Policy.**research QUESTIONS:**1. How Does Γ Affect the Optimal POLICY?2. Which Γ Values Lead to Faster CONVERGENCE?3. What Happens to State Values as Γ Changes?](#part-5-experiments-and-analysis-exercise-51-effect-of-discount-factor-γdefinitionthe-discount-factor-γ-determines-how-much-we-value-future-rewards-compared-to-immediate-rewardsmathematical-impactgt--rt1--γrt2--γ2rt3----sumk0infty-γk-rtk1interpretation-of-different-values--γ--0-only-immediate-rewards-matter-myopic-behavior--γ--1-all-future-rewards-equally-important-infinite-horizon--0--γ--1-future-rewards-are-discounted-realistictask-experiment-with-different-discount-factors-and-analyze-their-effect-on-the-optimal-policyresearch-questions1-how-does-γ-affect-the-optimal-policy2-which-γ-values-lead-to-faster-convergence3-what-happens-to-state-values-as-γ-changes)
    - [💰 Discount Factor Deep Dive: Balancing Present Vs Future**core Concept:** the Discount Factor Γ Controls the Agent's "patience" or Time Preference.---### ⏰ Time Value of Rewards**financial Analogy:**just like Money, Rewards Have "time Value":- $100 Today Vs $100 in 10 Years → Most Prefer Today (inflation, Uncertainty)- +10 Reward Now Vs +10 Reward in 100 Time Steps → Usually Prefer Immediate**mathematical Effect:**- **Γ = 0.1**: Reward 10 Steps Away Is Worth 0.1¹⁰ = 0.0000000001 of Current Reward- **Γ = 0.9**: Reward 10 Steps Away Is Worth 0.9¹⁰ = 0.35 of Current Reward- **Γ = 0.99**: Reward 10 Steps Away Is Worth 0.99¹⁰ = 0.90 of Current Reward---### 🌎 Real-world Analogies**γ = 0.1 (very Impatient/myopic):**- 🍕 "I Want Pizza Now, Don't Care About Health Consequences"- 💳 "BUY with Credit Card, Ignore Interest Charges"- 🚗 "take Fastest Route, Ignore Traffic Fines"**γ = 0.5 (moderately Patient):**- 🏃 "exercise Sometimes for Health Benefits"- 💰 "save Some Money, Spend Some Now"- 📚 "study When Motivated, Party When Not"**γ = 0.9 (balanced):**- 💪 "exercise Regularly for Long-term Health"- 🎓 "study Hard Now for Career Benefits Later"- 💰 "invest Consistently for Retirement"**γ = 0.99 (very Patient):**- 🌱 "plant Trees for Future Generations"- 🏠 "BUY House as Long-term Investment"- 🌍 "address Climate Change for Distant Future"---### 📊 Effect on Optimal Policy**low Γ (myopic Behavior):**- Takes Shortest Immediate Path to Reward- Ignores Long-term Consequences- May Get Stuck in Local Optima- Fast Convergence but Potentially Poor Solutions**high Γ (farsighted Behavior):**- Considers Long-term Consequences- May Take Longer Paths for Better Future Outcomes- Explores More Thoroughly- Slower Convergence but Better Final Solutions**in Gridworld Context:**- **low Γ**: Rushes toward Goal, Ignoring Obstacles- **high Γ**: Carefully Plans Path, Avoids Risky Moves#### **mathematical Impact**return Formula**: G*t = R*{T+1} + ΓR*{T+2} + Γ²R*{T+3} + Γ³R*{T+4} + ...**examples**:**γ = 0.9** (patient Agent):- G*t = R*{T+1} + 0.9×R*{T+2} + 0.81×R*{T+3} + 0.729×R*{T+4} + ...- Reward in 1 Step: Worth 100% of Immediate Reward- Reward in 2 Steps: Worth 90% of Immediate Reward - Reward in 3 Steps: Worth 81% of Immediate Reward- Reward in 10 Steps: Worth 35% of Immediate Reward**γ = 0.1** (impatient Agent):- G*t = R*{T+1} + 0.1×R*{T+2} + 0.01×R*{T+3} + 0.001×R_{T+4} + ...- Reward in 2 Steps: Worth Only 10% of Immediate Reward- Reward in 3 Steps: Worth Only 1% of Immediate Reward- Very Myopic - Only Cares About Next Few Steps#### **real-world Analogies**γ = 0.1** (very Impatient):- 🍕 "I Want Pizza Now, Don't Care About Health Consequences"- 📱 "BUY the Cheapest Phone, Ignore Long-term Durability" - 🚗 "take the Fastest Route, Ignore Traffic Fines"**γ = 0.9** (balanced):- 💪 "exercise Now for Health Benefits Later"- 🎓 "study Hard Now for Career Benefits Later"- 💰 "invest Money for Retirement"**γ = 0.99** (very Patient):- 🌱 "plant Trees for Future Generations"- 🏠 "BUY a House as Long-term Investment"- 🌍 "address Climate Change for Distant Future"#### **effect on Optimal Policy**low Γ (myopic Behavior)**:- Takes Shortest Path to Goal- Ignores Long-term Consequences - Might Take Dangerous Shortcuts- Policy: "rush to Goal, Avoid Obstacles Minimally"**high Γ (farsighted Behavior)**:- Takes Safer, Longer Paths- Values Long-term Safety- More Conservative Decisions- Policy: "GET to Goal Safely, Even If It Takes Longer"#### **choosing Γ in PRACTICE**CONSIDER**:1. **problem Horizon**: Short-term Tasks → Lower Γ, Long-term Tasks → Higher Γ2. **uncertainty**: More Uncertain Future → Lower Γ3. **safety**: Safety-critical Applications → Higher Γ4. **computational**: Higher Γ → Slower Convergence**common Values**:- **Γ = 0.9**: General Purpose, Good Balance- **Γ = 0.95-0.99**: Long-term Planning Tasks- **Γ = 0.1-0.5**: Short-term Reactive Tasks- **Γ = 1.0**: Infinite Horizon, Theoretical Studies (CAN Cause Issues)#### **debugging with Γ**if Your Agent:- **ignores Long-term Rewards**: Increase Γ- **IS Too Conservative**: Decrease Γ - **won't Converge**: Check If Γ Is Too Close to 1- **makes Random Decisions**: Γ Might Be Too Low](#-discount-factor-deep-dive-balancing-present-vs-futurecore-concept-the-discount-factor-γ-controls-the-agents-patience-or-time-preference-----time-value-of-rewardsfinancial-analogyjust-like-money-rewards-have-time-value--100-today-vs-100-in-10-years--most-prefer-today-inflation-uncertainty--10-reward-now-vs-10-reward-in-100-time-steps--usually-prefer-immediatemathematical-effect--γ--01-reward-10-steps-away-is-worth-01¹⁰--00000000001-of-current-reward--γ--09-reward-10-steps-away-is-worth-09¹⁰--035-of-current-reward--γ--099-reward-10-steps-away-is-worth-099¹⁰--090-of-current-reward-----real-world-analogiesγ--01-very-impatientmyopic---i-want-pizza-now-dont-care-about-health-consequences---buy-with-credit-card-ignore-interest-charges---take-fastest-route-ignore-traffic-finesγ--05-moderately-patient---exercise-sometimes-for-health-benefits---save-some-money-spend-some-now---study-when-motivated-party-when-notγ--09-balanced---exercise-regularly-for-long-term-health---study-hard-now-for-career-benefits-later---invest-consistently-for-retirementγ--099-very-patient---plant-trees-for-future-generations---buy-house-as-long-term-investment---address-climate-change-for-distant-future-----effect-on-optimal-policylow-γ-myopic-behavior--takes-shortest-immediate-path-to-reward--ignores-long-term-consequences--may-get-stuck-in-local-optima--fast-convergence-but-potentially-poor-solutionshigh-γ-farsighted-behavior--considers-long-term-consequences--may-take-longer-paths-for-better-future-outcomes--explores-more-thoroughly--slower-convergence-but-better-final-solutionsin-gridworld-context--low-γ-rushes-toward-goal-ignoring-obstacles--high-γ-carefully-plans-path-avoids-risky-moves-mathematical-impactreturn-formula-gt--rt1--γrt2--γ²rt3--γ³rt4--examplesγ--09-patient-agent--gt--rt1--09rt2--081rt3--0729rt4----reward-in-1-step-worth-100-of-immediate-reward--reward-in-2-steps-worth-90-of-immediate-reward---reward-in-3-steps-worth-81-of-immediate-reward--reward-in-10-steps-worth-35-of-immediate-rewardγ--01-impatient-agent--gt--rt1--01rt2--001rt3--0001r_t4----reward-in-2-steps-worth-only-10-of-immediate-reward--reward-in-3-steps-worth-only-1-of-immediate-reward--very-myopic---only-cares-about-next-few-steps-real-world-analogiesγ--01-very-impatient---i-want-pizza-now-dont-care-about-health-consequences---buy-the-cheapest-phone-ignore-long-term-durability----take-the-fastest-route-ignore-traffic-finesγ--09-balanced---exercise-now-for-health-benefits-later---study-hard-now-for-career-benefits-later---invest-money-for-retirementγ--099-very-patient---plant-trees-for-future-generations---buy-a-house-as-long-term-investment---address-climate-change-for-distant-future-effect-on-optimal-policylow-γ-myopic-behavior--takes-shortest-path-to-goal--ignores-long-term-consequences---might-take-dangerous-shortcuts--policy-rush-to-goal-avoid-obstacles-minimallyhigh-γ-farsighted-behavior--takes-safer-longer-paths--values-long-term-safety--more-conservative-decisions--policy-get-to-goal-safely-even-if-it-takes-longer-choosing-γ-in-practiceconsider1-problem-horizon-short-term-tasks--lower-γ-long-term-tasks--higher-γ2-uncertainty-more-uncertain-future--lower-γ3-safety-safety-critical-applications--higher-γ4-computational-higher-γ--slower-convergencecommon-values--γ--09-general-purpose-good-balance--γ--095-099-long-term-planning-tasks--γ--01-05-short-term-reactive-tasks--γ--10-infinite-horizon-theoretical-studies-can-cause-issues-debugging-with-γif-your-agent--ignores-long-term-rewards-increase-γ--is-too-conservative-decrease-γ---wont-converge-check-if-γ-is-too-close-to-1--makes-random-decisions-γ-might-be-too-low)
    - [Exercise 5.2: Modified Environment Experiments**task A**: Modify the Reward Structure and Analyze How It Affects the Optimal Policy:- Change Step Reward from -0.1 to -1.0 (higher Cost for Each Step)- Change Goal Reward from 10 to 5- Add Positive Rewards for Certain States**task B**: Experiment with Different Obstacle Configurations:- Remove Some Obstacles- Add More Obstacles- Change Obstacle Positions**task C**: Test with Different Starting Positions and Analyze Convergence.](#exercise-52-modified-environment-experimentstask-a-modify-the-reward-structure-and-analyze-how-it-affects-the-optimal-policy--change-step-reward-from--01-to--10-higher-cost-for-each-step--change-goal-reward-from-10-to-5--add-positive-rewards-for-certain-statestask-b-experiment-with-different-obstacle-configurations--remove-some-obstacles--add-more-obstacles--change-obstacle-positionstask-c-test-with-different-starting-positions-and-analyze-convergence)
  - [Part 6: Summary and Key Takeaways### What We've LEARNED**1. Markov Decision Processes (mdps):**- **framework**: Sequential Decision Making under Uncertainty- **components**: (S, A, P, R, Γ) - States, Actions, Transitions, Rewards, Discount- **markov Property**: Future Depends Only on Current State, Not History- **foundation**: Mathematical Basis for All Rl ALGORITHMS**2. Value Functions:**- **v^π(s)**: Expected Return Starting from State S Following Policy Π - **q^π(s,a)**: Expected Return Taking Action a in State S, Then Following Π- **relationship**: V^π(s) = Σ*a Π(a|s) Q^π(s,a)- **purpose**: Measure "goodness" of States and ACTIONS**3. Bellman Equations:**- **for V**: V^π(s) = Σ*a Π(a|s) Σ*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]- **for Q**: Q^π(s,a) = Σ*{s'} P(s'|s,a)[r(s,a,s') + Γ Σ*{a'} Π(a'|s')q^π(s',a')]- **significance**: Recursive Relationship Enabling Dynamic Programming SOLUTIONS**4. Policy Evaluation:**- **algorithm**: Iterative Method to Compute V^π Given Policy Π- **convergence**: Guaranteed for Finite Mdps with Γ < 1- **application**: Foundation for Policy Iteration and Value ITERATION**5. Policy Improvement:**- **theorem**: Greedy Policy W.r.t. V^π Is at Least as Good as Π- **formula**: Π'(s) = Argmax*a Q^π(s,a)- **monotonicity**: Each Improvement Step Yields Better or Equal POLICY**6. Policy Iteration:**- **algorithm**: Alternates between Evaluation and Improvement- **guarantee**: Converges to Optimal Policy Π*- **efficiency**: Usually Converges in Few Iterations---### Key Insights from Experiments**discount Factor (Γ) Effects:**- **low Γ**: Myopic Behavior, Focuses on Immediate Rewards- **high Γ**: Farsighted Behavior, Considers Long-term Consequences- **trade-off**: Convergence Speed Vs Solution Quality**environment Structure Impact:**- **reward Structure**: Significantly Affects Optimal Policy- **obstacles**: Create Navigation Challenges Requiring Planning- **starting Position**: Can Influence Learning Dynamics**algorithm Characteristics:**- **model-based**: Requires Knowledge of P(s'|s,a) and R(s,a,s')- **exact Solution**: Finds Truly Optimal Policy (unlike Approximate Methods)- **computational Cost**: Scales with State Space Size---### Connections to Advanced Topics**what This Enables:**- **value Iteration**: Direct Optimization of Value Function- **q-learning**: Model-free Learning of Action-value Functions- **deep Rl**: Neural Network Function Approximation- **policy Gradients**: Direct Policy Optimization Methods**next Steps in Learning:**- **temporal Difference Learning**: Learn from Incomplete Episodes- **function Approximation**: Handle Large/continuous State Spaces- **exploration Vs Exploitation**: Balance Learning and Performance- **multi-agent Systems**: Multiple Learning Agents Interacting---### Reflection Questions**theoretical UNDERSTANDING:**1. How Would Stochastic Transitions Affect the Optimal POLICY?2. What Happens with Continuous State or Action SPACES?3. How Do We Handle Unknown Environment DYNAMICS?4. What Are Computational Limits for Large State Spaces?**practical APPLICATIONS:**1. How Could You Apply Mdps to Real-world Decision PROBLEMS?2. What Modifications Would Be Needed for Competitive SCENARIOS?3. How Would You Handle Partially Observable ENVIRONMENTS?4. What Safety Considerations Are Important in Rl Applications?](#part-6-summary-and-key-takeaways-what-weve-learned1-markov-decision-processes-mdps--framework-sequential-decision-making-under-uncertainty--components-s-a-p-r-γ---states-actions-transitions-rewards-discount--markov-property-future-depends-only-on-current-state-not-history--foundation-mathematical-basis-for-all-rl-algorithms2-value-functions--vπs-expected-return-starting-from-state-s-following-policy-π---qπsa-expected-return-taking-action-a-in-state-s-then-following-π--relationship-vπs--σa-πas-qπsa--purpose-measure-goodness-of-states-and-actions3-bellman-equations--for-v-vπs--σa-πas-σs-pssarsas--γvπs--for-q-qπsa--σs-pssarsas--γ-σa-πasqπsa--significance-recursive-relationship-enabling-dynamic-programming-solutions4-policy-evaluation--algorithm-iterative-method-to-compute-vπ-given-policy-π--convergence-guaranteed-for-finite-mdps-with-γ--1--application-foundation-for-policy-iteration-and-value-iteration5-policy-improvement--theorem-greedy-policy-wrt-vπ-is-at-least-as-good-as-π--formula-πs--argmaxa-qπsa--monotonicity-each-improvement-step-yields-better-or-equal-policy6-policy-iteration--algorithm-alternates-between-evaluation-and-improvement--guarantee-converges-to-optimal-policy-π--efficiency-usually-converges-in-few-iterations----key-insights-from-experimentsdiscount-factor-γ-effects--low-γ-myopic-behavior-focuses-on-immediate-rewards--high-γ-farsighted-behavior-considers-long-term-consequences--trade-off-convergence-speed-vs-solution-qualityenvironment-structure-impact--reward-structure-significantly-affects-optimal-policy--obstacles-create-navigation-challenges-requiring-planning--starting-position-can-influence-learning-dynamicsalgorithm-characteristics--model-based-requires-knowledge-of-pssa-and-rsas--exact-solution-finds-truly-optimal-policy-unlike-approximate-methods--computational-cost-scales-with-state-space-size----connections-to-advanced-topicswhat-this-enables--value-iteration-direct-optimization-of-value-function--q-learning-model-free-learning-of-action-value-functions--deep-rl-neural-network-function-approximation--policy-gradients-direct-policy-optimization-methodsnext-steps-in-learning--temporal-difference-learning-learn-from-incomplete-episodes--function-approximation-handle-largecontinuous-state-spaces--exploration-vs-exploitation-balance-learning-and-performance--multi-agent-systems-multiple-learning-agents-interacting----reflection-questionstheoretical-understanding1-how-would-stochastic-transitions-affect-the-optimal-policy2-what-happens-with-continuous-state-or-action-spaces3-how-do-we-handle-unknown-environment-dynamics4-what-are-computational-limits-for-large-state-spacespractical-applications1-how-could-you-apply-mdps-to-real-world-decision-problems2-what-modifications-would-be-needed-for-competitive-scenarios3-how-would-you-handle-partially-observable-environments4-what-safety-considerations-are-important-in-rl-applications)
    - [🧠 Common Misconceptions and Intuitive Understandingbefore We Wrap Up, Let's Address Some Common Confusions and Solidify Understanding:#### **❌ Common MISCONCEPTIONS**1. "value Functions Are Just Rewards"**- ❌ Wrong: V(s) ≠ R(s) - ✅ Correct: V(s) = Expected Total Future Reward from State S- 🔍 Think: V(s) Is like Your Bank Account Balance, R(s) Is Your Daily INCOME**2. "q(s,a) Tells Me the Best Action"**- ❌ Wrong: Q(s,a) Is Not Binary Good/bad- ✅ Correct: Q(s,a) Is the Expected Value of Taking Action A- 🔍 Think: Compare Q-values to Choose Best Action: Argmax_a Q(S,A)**3. "policy Iteration Always Takes Many Steps"**- ❌ Wrong: Often Converges in 2-4 Iterations- ✅ Correct: Convergence Is Usually Very Fast- 🔍 Think: Once You Find a Good Strategy, Small Improvements Are ENOUGH**4. "random Policy Is Always Bad"**- ❌ Wrong: Random Policy Can Be Good for Exploration- ✅ Correct: Depends on Environment and Goals- 🔍 Think: Sometimes Trying New Things Leads to Better Discoveries#### **🎯 Key Intuitions to REMEMBER**1. the Big Picture Flow**:```environment → Policy → Actions → Rewards → Better Policy → REPEAT```**2. Value Functions as Gps**:- V(s): "HOW Good Is This Location Overall?"- Q(s,a): "HOW Good Is Taking This Road from This LOCATION?"**3. Bellman Equations as Consistency**:- "MY Value Should Equal Immediate Reward + Discounted Future Value"- Like: "MY Wealth = Today's Income + Tomorrow's WEALTH"**4. Policy Improvement as Learning**:- "IF I Know What Each Action Leads To, I Can Choose Better Actions"- Like: "IF I Know Exam Results for Each Study Method, I Can Study Better"#### **🔧 Troubleshooting Guide****if Values Don't Converge**:- Check If Γ < 1 - Reduce Convergence Threshold (theta)- Check for Bugs in Transition Probabilities**if Policy Doesn't Improve**:- Environment Might Be Too Simple (already Optimal)- Check Reward Structure - Might Need More Differentiation- Verify Policy Improvement Logic**if Results Seem Weird**:- Visualize Value Functions and Policies- Start with Simpler Environment- Check Reward Signs (positive/negative)#### **🚀 Connecting to Future Topics**what We Learned Here Enables:- **value Iteration**: Direct Value Optimization (next Week!)- **q-learning**: Learn Q-values without Knowing the Model- **deep Rl**: Use Neural Networks to Handle Large State Spaces- **policy Gradients**: Directly Optimize the Policy Parameters#### **🎭 the Rl Mindset**think like an Rl AGENT:1. **observe** Your Current Situation (STATE)2. **consider** Your Options (actions) 3. **predict** Outcomes (USE Your MODEL/EXPERIENCE)4. **choose** the Best Option (POLICY)5. **learn** from Results (update VALUES/POLICY)6. **repeat** until Masterythis Mindset Applies To:- Career Decisions- Investment Choices - Game Strategies- Daily Life Optimization](#-common-misconceptions-and-intuitive-understandingbefore-we-wrap-up-lets-address-some-common-confusions-and-solidify-understanding--common-misconceptions1-value-functions-are-just-rewards---wrong-vs--rs----correct-vs--expected-total-future-reward-from-state-s---think-vs-is-like-your-bank-account-balance-rs-is-your-daily-income2-qsa-tells-me-the-best-action---wrong-qsa-is-not-binary-goodbad---correct-qsa-is-the-expected-value-of-taking-action-a---think-compare-q-values-to-choose-best-action-argmax_a-qsa3-policy-iteration-always-takes-many-steps---wrong-often-converges-in-2-4-iterations---correct-convergence-is-usually-very-fast---think-once-you-find-a-good-strategy-small-improvements-are-enough4-random-policy-is-always-bad---wrong-random-policy-can-be-good-for-exploration---correct-depends-on-environment-and-goals---think-sometimes-trying-new-things-leads-to-better-discoveries--key-intuitions-to-remember1-the-big-picture-flowenvironment--policy--actions--rewards--better-policy--repeat2-value-functions-as-gps--vs-how-good-is-this-location-overall--qsa-how-good-is-taking-this-road-from-this-location3-bellman-equations-as-consistency--my-value-should-equal-immediate-reward--discounted-future-value--like-my-wealth--todays-income--tomorrows-wealth4-policy-improvement-as-learning--if-i-know-what-each-action-leads-to-i-can-choose-better-actions--like-if-i-know-exam-results-for-each-study-method-i-can-study-better--troubleshooting-guideif-values-dont-converge--check-if-γ--1---reduce-convergence-threshold-theta--check-for-bugs-in-transition-probabilitiesif-policy-doesnt-improve--environment-might-be-too-simple-already-optimal--check-reward-structure---might-need-more-differentiation--verify-policy-improvement-logicif-results-seem-weird--visualize-value-functions-and-policies--start-with-simpler-environment--check-reward-signs-positivenegative--connecting-to-future-topicswhat-we-learned-here-enables--value-iteration-direct-value-optimization-next-week--q-learning-learn-q-values-without-knowing-the-model--deep-rl-use-neural-networks-to-handle-large-state-spaces--policy-gradients-directly-optimize-the-policy-parameters--the-rl-mindsetthink-like-an-rl-agent1-observe-your-current-situation-state2-consider-your-options-actions-3-predict-outcomes-use-your-modelexperience4-choose-the-best-option-policy5-learn-from-results-update-valuespolicy6-repeat-until-masterythis-mindset-applies-to--career-decisions--investment-choices---game-strategies--daily-life-optimization)


# Table of Contents

- [Deep Reinforcement Learning - Session 2 Exercise
#
# Markov Decision Processes and Value Functions**objective**: This Comprehensive Exercise Covers Fundamental Concepts of Reinforcement Learning Including Markov Decision Processes (mdps), Value Functions, Bellman Equations, and Policy Evaluation Methods.
#
#
# Topics COVERED:1. Introduction to Reinforcement Learning FRAMEWORK2. Markov Decision Processes (MDPS)3. Value Functions (state-value and ACTION-VALUE)4. Bellman EQUATIONS5. Policy Evaluation and IMPROVEMENT6. Practical Implementation with Gridworld Environment
#
#
# Learning Outcomes:by the End of This Exercise, You Will Understand:- the Mathematical Foundation of Mdps- How to Compute Value Functions- the Relationship between Policies and Value Functions- Implementation of Basic Rl Algorithms](
#deep-reinforcement-learning---session-2-exercise-markov-decision-processes-and-value-functionsobjective-this-comprehensive-exercise-covers-fundamental-concepts-of-reinforcement-learning-including-markov-decision-processes-mdps-value-functions-bellman-equations-and-policy-evaluation-methods-topics-covered1-introduction-to-reinforcement-learning-framework2-markov-decision-processes-mdps3-value-functions-state-value-and-action-value4-bellman-equations5-policy-evaluation-and-improvement6-practical-implementation-with-gridworld-environment-learning-outcomesby-the-end-of-this-exercise-you-will-understand--the-mathematical-foundation-of-mdps--how-to-compute-value-functions--the-relationship-between-policies-and-value-functions--implementation-of-basic-rl-algorithms)
- [Part 1: Theoretical Foundation
#
#
# 1.1 Reinforcement Learning Framework**definition:**reinforcement Learning Is a Computational Approach to Learning from Interaction. the Key Elements Are:- **agent**: the Learner and Decision Maker - the Entity That Makes Choices- **environment**: the World the Agent Interacts with - Everything outside the Agent- **state (s)**: Current Situation of the Agent - Describes the Current Circumstances- **action (a)**: Choices Available to the Agent - Decisions That Can Be Made- **reward (r)**: Numerical Feedback from Environment - Immediate Feedback Signal- **policy (π)**: Agent's Strategy for Choosing Actions - Mapping from States to Actions**real-world Analogy:**think of Rl like Learning to Drive:- **agent** = the Driver (you)- **environment** = Roads, Traffic, Weather Conditions- **state** = Current Speed, Position, Traffic around You- **actions** = Accelerate, Brake, Turn Left/right- **reward** = Positive for Safe Driving, Negative for Accidents- **policy** = Your Driving Strategy (cautious, Aggressive, Etc.)---
#
#
# 1.2 Markov Decision Process (mdp)**definition:**an Mdp Is Defined by the Tuple (S, A, P, R, Γ) Where:- **s**: Set of States - All Possible Situations the Agent Can Encounter- **a**: Set of Actions - All Possible Decisions Available to the Agent- **p**: Transition Probability Function P(s'|s,a) - Probability of Moving to State S' Given Current State S and Action A- **r**: Reward Function R(s,a,s') - Immediate Reward Received for Transitioning from S to S' Via Action A- **γ**: Discount Factor (0 ≤ Γ ≤ 1) - Determines Importance of Future Rewards**markov Property:**the Future Depends Only on the Current State, Not on the History of How We Got There. MATHEMATICALLY:P(S*{T+1} = S' | S*t = S, A*t = A, S*{T-1}, A*{T-1}, ..., S*0, A*0) = P(S*{T+1} = S' | S*t = S, A*t = A)**intuition:**the Current State Contains All Information Needed to Make Optimal Decisions. the past Is Already "encoded" in the Current State.---
#
#
# 1.3 Value Functions**state-value Function:**$$v^π(s) = \mathbb{e}*π[g*t | S_t = S]$$**interpretation:** Expected Total Reward When Starting from State S and Following Policy Π. It Answers: "HOW Good Is It to Be in This State?"**action-value Function:**$$q^π(s,a) = \mathbb{e}*π[g*t | S*t = S, A*t = A]$$**interpretation:** Expected Total Reward When Taking Action a in State S and Then Following Policy Π. It Answers: "HOW Good Is It to Take This Specific Action in This State?"**return (cumulative Reward):**$$g*t = R*{T+1} + ΓR*{T+2} + Γ^2R*{T+3} + ... = \SUM*{K=0}^{\INFTY} Γ^k R*{T+K+1}$$**WHY Discount Factor Γ?**- **Γ = 0**: Only Immediate Rewards Matter (myopic)- **Γ = 1**: All Future Rewards Are Equally Important- **0 < Γ < 1**: Future Rewards Are Discounted (realistic for Most Scenarios)---
#
#
# 1.4 Bellman Equations**bellman Equation for State-value Function:**$$v^π(s) = \sum*a Π(a|s) \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**interpretation:** the Value of a State Equals the Immediate Reward Plus the Discounted Value of the Next State, Averaged over All Possible Actions and Transitions.**bellman Equation for Action-value Function:**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γ \sum*{a'} Π(a'|s')q^π(s',a')]$$**key Insight:** the Bellman Equations Express a Recursive Relationship - the Value of a State (OR State-action Pair) Depends on the Immediate Reward Plus the Discounted Value of Future States. This Is the Mathematical Foundation for Most Rl Algorithms.](
#part-1-theoretical-foundation-11-reinforcement-learning-frameworkdefinitionreinforcement-learning-is-a-computational-approach-to-learning-from-interaction-the-key-elements-are--agent-the-learner-and-decision-maker---the-entity-that-makes-choices--environment-the-world-the-agent-interacts-with---everything-outside-the-agent--state-s-current-situation-of-the-agent---describes-the-current-circumstances--action-a-choices-available-to-the-agent---decisions-that-can-be-made--reward-r-numerical-feedback-from-environment---immediate-feedback-signal--policy-π-agents-strategy-for-choosing-actions---mapping-from-states-to-actionsreal-world-analogythink-of-rl-like-learning-to-drive--agent--the-driver-you--environment--roads-traffic-weather-conditions--state--current-speed-position-traffic-around-you--actions--accelerate-brake-turn-leftright--reward--positive-for-safe-driving-negative-for-accidents--policy--your-driving-strategy-cautious-aggressive-etc----12-markov-decision-process-mdpdefinitionan-mdp-is-defined-by-the-tuple-s-a-p-r-γ-where--s-set-of-states---all-possible-situations-the-agent-can-encounter--a-set-of-actions---all-possible-decisions-available-to-the-agent--p-transition-probability-function-pssa---probability-of-moving-to-state-s-given-current-state-s-and-action-a--r-reward-function-rsas---immediate-reward-received-for-transitioning-from-s-to-s-via-action-a--γ-discount-factor-0--γ--1---determines-importance-of-future-rewardsmarkov-propertythe-future-depends-only-on-the-current-state-not-on-the-history-of-how-we-got-there-mathematicallypst1--s--st--s-at--a-st-1-at-1--s0-a0--pst1--s--st--s-at--aintuitionthe-current-state-contains-all-information-needed-to-make-optimal-decisions-the-past-is-already-encoded-in-the-current-state----13-value-functionsstate-value-functionvπs--mathbbeπgt--s_t--sinterpretation-expected-total-reward-when-starting-from-state-s-and-following-policy-π-it-answers-how-good-is-it-to-be-in-this-stateaction-value-functionqπsa--mathbbeπgt--st--s-at--ainterpretation-expected-total-reward-when-taking-action-a-in-state-s-and-then-following-policy-π-it-answers-how-good-is-it-to-take-this-specific-action-in-this-statereturn-cumulative-rewardgt--rt1--γrt2--γ2rt3----sumk0infty-γk-rtk1why-discount-factor-γ--γ--0-only-immediate-rewards-matter-myopic--γ--1-all-future-rewards-are-equally-important--0--γ--1-future-rewards-are-discounted-realistic-for-most-scenarios----14-bellman-equationsbellman-equation-for-state-value-functionvπs--suma-πas-sums-pssarsas--γvπsinterpretation-the-value-of-a-state-equals-the-immediate-reward-plus-the-discounted-value-of-the-next-state-averaged-over-all-possible-actions-and-transitionsbellman-equation-for-action-value-functionqπsa--sums-pssarsas--γ-suma-πasqπsakey-insight-the-bellman-equations-express-a-recursive-relationship---the-value-of-a-state-or-state-action-pair-depends-on-the-immediate-reward-plus-the-discounted-value-of-future-states-this-is-the-mathematical-foundation-for-most-rl-algorithms)
- [📚 Common Misconceptions and Clarifications**misconception 1: "THE Agent Knows the Environment Model"**reality:** in Most Rl Problems, the Agent Doesn't Know P(s'|s,a) or R(s,a,s'). This Is Called "model-free" Rl, Where the Agent Learns through Trial and Error.**misconception 2: "higher Rewards Are Always Better"**reality:** the Goal Is to Maximize *cumulative* Reward, Not Immediate Reward. Sometimes Taking a Small Immediate Reward Prevents Getting a Much Larger Future Reward.**misconception 3: "THE Policy Should Always Be Deterministic"****reality:** Stochastic Policies (that Output Probabilities) Are Often Better Because They Allow for Exploration and Can Be Optimal in Certain Environments.---
#
#
# 🧠 Building Intuition: Restaurant Example**scenario:** You're Choosing Restaurants to Visit in a New City.**mdp Components:**- **states**: Your Hunger Level, Location, Time of Day, Budget- **actions**: Choose Restaurant A, B, C, or Cook at Home- **rewards**: Satisfaction from Food (immediate) + Health Effects (long-term)- **transitions**: How Your State Changes after Eating**value Functions:**- **v(hungry, Downtown, Evening)**: How Good Is This Situation Overall?- **q(hungry, Downtown, Evening, "restaurant A")**: How Good Is Choosing Restaurant a in This Situation?**policy Learning:** Initially Random Choices → Gradually Prefer Restaurants That Gave Good Experiences → Eventually Develop a Strategy That Considers Health, Taste, Cost, and Convenience.---
#
#
# 🔧 Mathematical Properties and Theorems**theorem 1: Existence and Uniqueness of Value Functions**for Any Policy Π and Finite Mdp, There Exists a Unique Solution to the Bellman Equations.**theorem 2: Bellman Optimality Principle**a Policy Π Is Optimal If and Only If:$$v^π(s) = \max_a Q^π(s,a) \text{ for All } S \IN S$$**theorem 3: Policy Improvement Theorem**if Π' Is Greedy with Respect to V^π, Then V^π'(s) ≥ V^π(s) for All States S.**practical Implications:**- We Can Always Improve a Policy by Being Greedy with Respect to Its Value Function- There Always Exists an Optimal Policy (MAY Not Be Unique)- the Optimal Value Function Satisfies the Bellman Optimality Equations- **rewards**: +1 for Safe Driving, -10 for Accidents, -1 for Speeding Tickets- **policy**: Your Driving Strategy (aggressive, Conservative, Etc.)
#
#
#
# **why Markov Property Matters**the **markov Property** Means "THE Future Depends Only on the Present, Not the Past."**example**: in Chess, to Decide Your Next Move, You Only Need to See the Current Board Position. You Don't Need to Know How the Pieces Got There - the Complete Game History Is Irrelevant for Making the Optimal Next Move.**non-markov Example**: Predicting Tomorrow's Weather Based Only on Today's Weather (YOU Need Historical Patterns).
#
#
#
# **understanding the Discount Factor (γ)**the Discount Factor Determines How Much You Care About Future Rewards:- **Γ = 0**: "I Only Care About Immediate Rewards" (very Myopic)- Example: Only Caring About This Month's Salary, Not Career Growth - **Γ = 0.9**: "future Rewards Are Worth 90% of Immediate Rewards"- Example: Investing Money - You Value Future Returns but Prefer Sooner - **Γ = 1**: "future Rewards Are as Valuable as Immediate Rewards"- Example: Climate Change Actions - Long-term Benefits Matter Equally**mathematical Impact**:- Return G*t = R*{T+1} + ΓR*{T+2} + Γ²R*{T+3} + ...- with Γ=0.9: G*t = R*{T+1} + 0.9×R*{T+2} + 0.81×R*{T+3} + ...- Future Rewards Get Progressively Less Important](
#-common-misconceptions-and-clarificationsmisconception-1-the-agent-knows-the-environment-modelreality-in-most-rl-problems-the-agent-doesnt-know-pssa-or-rsas-this-is-called-model-free-rl-where-the-agent-learns-through-trial-and-errormisconception-2-higher-rewards-are-always-betterreality-the-goal-is-to-maximize-cumulative-reward-not-immediate-reward-sometimes-taking-a-small-immediate-reward-prevents-getting-a-much-larger-future-rewardmisconception-3-the-policy-should-always-be-deterministicreality-stochastic-policies-that-output-probabilities-are-often-better-because-they-allow-for-exploration-and-can-be-optimal-in-certain-environments-----building-intuition-restaurant-examplescenario-youre-choosing-restaurants-to-visit-in-a-new-citymdp-components--states-your-hunger-level-location-time-of-day-budget--actions-choose-restaurant-a-b-c-or-cook-at-home--rewards-satisfaction-from-food-immediate--health-effects-long-term--transitions-how-your-state-changes-after-eatingvalue-functions--vhungry-downtown-evening-how-good-is-this-situation-overall--qhungry-downtown-evening-restaurant-a-how-good-is-choosing-restaurant-a-in-this-situationpolicy-learning-initially-random-choices--gradually-prefer-restaurants-that-gave-good-experiences--eventually-develop-a-strategy-that-considers-health-taste-cost-and-convenience-----mathematical-properties-and-theoremstheorem-1-existence-and-uniqueness-of-value-functionsfor-any-policy-π-and-finite-mdp-there-exists-a-unique-solution-to-the-bellman-equationstheorem-2-bellman-optimality-principlea-policy-π-is-optimal-if-and-only-ifvπs--max_a-qπsa-text-for-all--s-in-stheorem-3-policy-improvement-theoremif-π-is-greedy-with-respect-to-vπ-then-vπs--vπs-for-all-states-spractical-implications--we-can-always-improve-a-policy-by-being-greedy-with-respect-to-its-value-function--there-always-exists-an-optimal-policy-may-not-be-unique--the-optimal-value-function-satisfies-the-bellman-optimality-equations--rewards-1-for-safe-driving--10-for-accidents--1-for-speeding-tickets--policy-your-driving-strategy-aggressive-conservative-etc-why-markov-property-mattersthe-markov-property-means-the-future-depends-only-on-the-present-not-the-pastexample-in-chess-to-decide-your-next-move-you-only-need-to-see-the-current-board-position-you-dont-need-to-know-how-the-pieces-got-there---the-complete-game-history-is-irrelevant-for-making-the-optimal-next-movenon-markov-example-predicting-tomorrows-weather-based-only-on-todays-weather-you-need-historical-patterns-understanding-the-discount-factor-γthe-discount-factor-determines-how-much-you-care-about-future-rewards--γ--0-i-only-care-about-immediate-rewards-very-myopic--example-only-caring-about-this-months-salary-not-career-growth---γ--09-future-rewards-are-worth-90-of-immediate-rewards--example-investing-money---you-value-future-returns-but-prefer-sooner---γ--1-future-rewards-are-as-valuable-as-immediate-rewards--example-climate-change-actions---long-term-benefits-matter-equallymathematical-impact--return-gt--rt1--γrt2--γ²rt3----with-γ09-gt--rt1--09rt2--081rt3----future-rewards-get-progressively-less-important)
- [🎮 Understanding Our Gridworld Environmentbefore We Dive into the Code, Let's Understand What We're Building:
#
#
#
# **the Gridworld SETUP**```(0,0) → → → (0,3) ↓ X X ↓ ↓ X ◯ ↓ (3,0) → → → (3,3) 🎯```**legend:**- `S` at (0,0): Starting Position- `🎯` at (3,3): Goal (treasure!)- `X`: Obstacles (walls or Pits)- `◯`: Regular Empty Spaces- Arrows: Possible Movements
#
#
#
# **why This Environment Is Perfect for LEARNING**1. **small & Manageable**: 4×4 Grid = 16 States (easy to VISUALIZE)2. **clear Objective**: Get from Start to GOAL3. **interesting Obstacles**: Forces Strategic THINKING4. **deterministic**: Same Action Always Leads to Same Result (FOR Now)
#
#
#
# **reward Structure Explained**- **goal Reward (+10)**: Big Positive Reward for Reaching the Treasure- **step Penalty (-0.1)**: Small Negative Reward for Each Move (encourages Efficiency)- **obstacle Penalty (-5)**: Big Negative Reward for Hitting Obstacles (safety First!)**why These Specific Values?**- Goal Reward Is Much Larger Than Step Penalty → Encourages Reaching the Goal- Obstacle Penalty Is Significant → Discourages Dangerous Moves- Step Penalty Is Small → Prevents Infinite Wandering without Being Too Harsh
#
#
#
# **state Representation**each State Is a Tuple (row, Column):- (0,0) = Top-left Corner- (3,3) = Bottom-right Corner - States Are like Gps Coordinates for Our Agent](
#-understanding-our-gridworld-environmentbefore-we-dive-into-the-code-lets-understand-what-were-building-the-gridworld-setup00----03--x-x---x---30----33-legend--s-at-00-starting-position---at-33-goal-treasure--x-obstacles-walls-or-pits---regular-empty-spaces--arrows-possible-movements-why-this-environment-is-perfect-for-learning1-small--manageable-44-grid--16-states-easy-to-visualize2-clear-objective-get-from-start-to-goal3-interesting-obstacles-forces-strategic-thinking4-deterministic-same-action-always-leads-to-same-result-for-now-reward-structure-explained--goal-reward-10-big-positive-reward-for-reaching-the-treasure--step-penalty--01-small-negative-reward-for-each-move-encourages-efficiency--obstacle-penalty--5-big-negative-reward-for-hitting-obstacles-safety-firstwhy-these-specific-values--goal-reward-is-much-larger-than-step-penalty--encourages-reaching-the-goal--obstacle-penalty-is-significant--discourages-dangerous-moves--step-penalty-is-small--prevents-infinite-wandering-without-being-too-harsh-state-representationeach-state-is-a-tuple-row-column--00--top-left-corner--33--bottom-right-corner---states-are-like-gps-coordinates-for-our-agent)
- [Part 2: Policy Definition and Evaluation
#
#
# Exercise 2.1: Define Different Policies**definition:**a Policy Π(a|s) Defines the Probability of Taking Action a in State S. It's the Agent's Strategy for Choosing Actions.**mathematical Representation:**$$\pi(a|s) = P(\text{action} = a | \text{state} = S)$$**types of Policies:**- **deterministic Policy**: Π(a|s) ∈ {0, 1} - Always Chooses the Same Action in a Given State- **stochastic Policy**: Π(a|s) ∈ [0, 1] - Chooses Actions Probabilistically**policies We'll IMPLEMENT:**1. **random Policy**: Equal Probability for All Valid ACTIONS2. **greedy Policy**: Always Move towards the Goal 3. **custom Policy**: Your Own Strategic Policy---
#
#
# Exercise 2.2: Policy Evaluation**definition:**policy Evaluation Computes the Value Function V^π(s) for a Given Policy Π. It Answers: "HOW Good Is This Policy?"**iterative Policy Evaluation ALGORITHM:**1. **initialize**: V(s) = 0 for All States S2. **repeat until Convergence**:- for Each State S:- V*new(s) = Σ*a Π(a|s) Σ*{s'} P(s'|s,a)[r(s,a,s') + ΓV(S')]3. **return**: Converged Value Function V**convergence Condition:**max*s |v*new(s) - V*old(s)| < Θ (where Θ Is a Small Threshold, E.g., 1E-6)**INTUITION:**WE Start with All State Values at Zero and Iteratively Update Them Based on the Bellman Equation until They Stabilize. It's like Repeatedly Asking "IF I Follow This Policy, How Much Reward Will I Get?" until the Answer Stops Changing.](
#part-2-policy-definition-and-evaluation-exercise-21-define-different-policiesdefinitiona-policy-πas-defines-the-probability-of-taking-action-a-in-state-s-its-the-agents-strategy-for-choosing-actionsmathematical-representationpias--ptextaction--a--textstate--stypes-of-policies--deterministic-policy-πas--0-1---always-chooses-the-same-action-in-a-given-state--stochastic-policy-πas--0-1---chooses-actions-probabilisticallypolicies-well-implement1-random-policy-equal-probability-for-all-valid-actions2-greedy-policy-always-move-towards-the-goal-3-custom-policy-your-own-strategic-policy----exercise-22-policy-evaluationdefinitionpolicy-evaluation-computes-the-value-function-vπs-for-a-given-policy-π-it-answers-how-good-is-this-policyiterative-policy-evaluation-algorithm1-initialize-vs--0-for-all-states-s2-repeat-until-convergence--for-each-state-s--vnews--σa-πas-σs-pssarsas--γvs3-return-converged-value-function-vconvergence-conditionmaxs-vnews---volds--θ-where-θ-is-a-small-threshold-eg-1e-6intuitionwe-start-with-all-state-values-at-zero-and-iteratively-update-them-based-on-the-bellman-equation-until-they-stabilize-its-like-repeatedly-asking-if-i-follow-this-policy-how-much-reward-will-i-get-until-the-answer-stops-changing)
- [🧭 Policy Deep Dive: Understanding Different Strategies**what Is a Policy?**a Policy Is like a Gps Navigation System for Our Agent. It Tells the Agent What to Do in Every Possible Situation.**mathematical Definition:**π(a|s) = Probability of Taking Action a When in State S---
#
#
# 📋 Types of Policies We'll IMPLEMENT**1. Random Policy** 🎲**strategy:** "when in Doubt, Flip a Coin"**mathematical Definition:** Π(a|s) = 1/|VALID_ACTIONS| for All Valid Actions**example:** at State (1,0), If We Can Go [UP, Down, Right], Each Has 33.33% Probability**advantages:**- Explores All Possibilities Equally- Simple to Implement- Guarantees Exploration**disadvantages:**- Not Very Efficient- like Wandering Randomly in a Maze- No Learning from EXPERIENCE---**2. Greedy Policy** 🎯**strategy:** "always Move Closer to the Goal"**mathematical Definition:** Π(a|s) = 1 If a Minimizes Distance to Goal, 0 Otherwise**example:** at State (1,0), If Goal Is at (3,3), Prefer "down" and "right"**advantages:**- Very Efficient When It Works- Direct Path to Goal- Fast Convergence**disadvantages:**- Can Get Stuck in Local Optima- Might Walk into Obstacles- No Exploration of Alternative PATHS---**3. Custom Policy** 🎨**strategy:** Your Creative Combination of Strategies**examples:**- **epsilon-greedy**: 90% Greedy, 10% Random- **safety-first**: Avoid Actions That Lead near Obstacles- **wall-follower**: Stay Close to Boundaries---
#
#
# 🎮 Real-world Analogies**policy Vs Strategy in Games:**think of Different Video Game Playing Styles:- **aggressive Player**: Always Attacks (deterministic Policy)- **defensive Player**: Always Defends (deterministic Policy)- **adaptive Player**: 70% Attack, 30% Defend (stochastic Policy)**why Stochastic Policies?**sometimes Randomness Helps:- **exploration**: Discover New Paths You Wouldn't Normally Try- **unpredictability**: in Competitive Games, Being Predictable Is Bad- **robustness**: Handle Uncertainty in the Environment**restaurant Choice Analogy:**- **random Policy**: Pick Restaurants Randomly- **greedy Policy**: Always Go to Your Current Favorite- **epsilon-greedy Policy**: Usually Go to Favorite, Sometimes Try Something New](
#-policy-deep-dive-understanding-different-strategieswhat-is-a-policya-policy-is-like-a-gps-navigation-system-for-our-agent-it-tells-the-agent-what-to-do-in-every-possible-situationmathematical-definitionπas--probability-of-taking-action-a-when-in-state-s-----types-of-policies-well-implement1-random-policy-strategy-when-in-doubt-flip-a-coinmathematical-definition-πas--1valid_actions-for-all-valid-actionsexample-at-state-10-if-we-can-go-up-down-right-each-has-3333-probabilityadvantages--explores-all-possibilities-equally--simple-to-implement--guarantees-explorationdisadvantages--not-very-efficient--like-wandering-randomly-in-a-maze--no-learning-from-experience---2-greedy-policy-strategy-always-move-closer-to-the-goalmathematical-definition-πas--1-if-a-minimizes-distance-to-goal-0-otherwiseexample-at-state-10-if-goal-is-at-33-prefer-down-and-rightadvantages--very-efficient-when-it-works--direct-path-to-goal--fast-convergencedisadvantages--can-get-stuck-in-local-optima--might-walk-into-obstacles--no-exploration-of-alternative-paths---3-custom-policy-strategy-your-creative-combination-of-strategiesexamples--epsilon-greedy-90-greedy-10-random--safety-first-avoid-actions-that-lead-near-obstacles--wall-follower-stay-close-to-boundaries-----real-world-analogiespolicy-vs-strategy-in-gamesthink-of-different-video-game-playing-styles--aggressive-player-always-attacks-deterministic-policy--defensive-player-always-defends-deterministic-policy--adaptive-player-70-attack-30-defend-stochastic-policywhy-stochastic-policiessometimes-randomness-helps--exploration-discover-new-paths-you-wouldnt-normally-try--unpredictability-in-competitive-games-being-predictable-is-bad--robustness-handle-uncertainty-in-the-environmentrestaurant-choice-analogy--random-policy-pick-restaurants-randomly--greedy-policy-always-go-to-your-current-favorite--epsilon-greedy-policy-usually-go-to-favorite-sometimes-try-something-new)
- [🔍 Understanding Policy Evaluation Step-by-steppolicy Evaluation Answers the Question: **"how Good Is Each State If I Follow This Policy?"**
#
#
#
# **the Intuition**imagine You're Evaluating Different Starting Positions in a Board Game:- Some Positions Are Naturally Better (closer to Winning)- Some Positions Are Worse (closer to Losing) - the "value" of a Position Depends on How Well You'll Do from There
#
#
#
# **mathematical Breakdown****the Bellman Equation for State Values:**```v^π(s) = Σ*a Π(a|s) × Σ*{s'} P(s'|s,a) × [r(s,a,s') + Γ × V^π(s')]```**let's Decode This Step by STEP:**1. **for Each Possible Action A**: Π(a|s) = "HOW Likely Am I to Take Action a in State S?"2. **for Each Possible Next State S'**: P(s'|s,a) = "IF I Take Action A, What's the Chance I End Up in State S'?"3. **calculate Immediate Reward + Future Value**: R(s,a,s') + Γ × V^π(s')- R(s,a,s') = "what Reward Do I Get Immediately?"- Γ × V^π(s') = "what's the Discounted Future VALUE?"4. **sum Everything Up**: This Gives the Expected Value of Being in State S
#
#
#
# **simple Example**let's Say We're at State (2,2) with a Random Policy:```python
# Random Policy: Equal Probability for All Valid Actionsπ(up|s) = 0.25, Π(down|s) = 0.25, Π(left|s) = 0.25, Π(right|s) = 0.25
# FOR Action "UP" → Next State (1,2)CONTRIBUTION*UP = 0.25 × 1.0 × (-0.1 + 0.9 × V(1,2))
# FOR Action "down" → Next State (3,2)CONTRIBUTION*DOWN = 0.25 × 1.0 × (-0.1 + 0.9 × V(3,2))
# ... and So on for Left and RIGHTV(2,2) = Contribution*up + Contribution*down + Contribution*left + Contribution*right```
#
#
#
# **why Iterative?**- We Start with V(s) = 0 for All States (initial Guess)- Each Iteration Improves Our Estimate Using Current Values- Eventually, Values Converge to True Values- like Asking "IF I Knew the Value of My Neighbors, What Would My Value Be?"
#
#
#
# **convergence Intuition**think of It like Gossip Spreading in a Neighborhood:- Initially, Nobody Knows the True "gossip" (values)- Each Iteration, Neighbors Share Information - Eventually, Everyone Converges to the Same True Story](
#-understanding-policy-evaluation-step-by-steppolicy-evaluation-answers-the-question-how-good-is-each-state-if-i-follow-this-policy-the-intuitionimagine-youre-evaluating-different-starting-positions-in-a-board-game--some-positions-are-naturally-better-closer-to-winning--some-positions-are-worse-closer-to-losing---the-value-of-a-position-depends-on-how-well-youll-do-from-there-mathematical-breakdownthe-bellman-equation-for-state-valuesvπs--σa-πas--σs-pssa--rsas--γ--vπslets-decode-this-step-by-step1-for-each-possible-action-a-πas--how-likely-am-i-to-take-action-a-in-state-s2-for-each-possible-next-state-s-pssa--if-i-take-action-a-whats-the-chance-i-end-up-in-state-s3-calculate-immediate-reward--future-value-rsas--γ--vπs--rsas--what-reward-do-i-get-immediately--γ--vπs--whats-the-discounted-future-value4-sum-everything-up-this-gives-the-expected-value-of-being-in-state-s-simple-examplelets-say-were-at-state-22-with-a-random-policypython-random-policy-equal-probability-for-all-valid-actionsπups--025-πdowns--025-πlefts--025-πrights--025-for-action-up--next-state-12contributionup--025--10---01--09--v12-for-action-down--next-state-32contributiondown--025--10---01--09--v32--and-so-on-for-left-and-rightv22--contributionup--contributiondown--contributionleft--contributionright-why-iterative--we-start-with-vs--0-for-all-states-initial-guess--each-iteration-improves-our-estimate-using-current-values--eventually-values-converge-to-true-values--like-asking-if-i-knew-the-value-of-my-neighbors-what-would-my-value-be-convergence-intuitionthink-of-it-like-gossip-spreading-in-a-neighborhood--initially-nobody-knows-the-true-gossip-values--each-iteration-neighbors-share-information---eventually-everyone-converges-to-the-same-true-story)
- [Exercise 2.3: Create Your Custom Policy**task**: Design and Implement Your Own Policy. Consider Strategies Like:- **wall-following**: Try to Stay Close to Walls- **risk-averse**: Avoid Obstacles with Higher Probability- **exploration-focused**: Balance between Moving towards Goal and Exploring**your Implementation Below**:](
#exercise-23-create-your-custom-policytask-design-and-implement-your-own-policy-consider-strategies-like--wall-following-try-to-stay-close-to-walls--risk-averse-avoid-obstacles-with-higher-probability--exploration-focused-balance-between-moving-towards-goal-and-exploringyour-implementation-below)
- [Part 3: Action-value Functions (q-functions)
#
#
# Exercise 3.1: Computing Q-values**definition:**the Action-value Function Q^π(s,a) Represents the Expected Return When Taking Action a in State S and Then Following Policy Π.**key Question Q-functions Answer:**q-functions Answer: "what If I Take This Specific Action Here, Then Follow My Policy?"**mathematical Relationships:****v from Q (policy-weighted Average):**$$v^π(s) = \sum*a Π(a|s) Q^π(s,a)$$**q from V (bellman Backup):**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**bellman Equation for Q:**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γ \sum*{a'} Π(a'|s')q^π(s',a')]$$**intuition:**- **v(s)**: "HOW Good Is This State?" (following Current Policy)- **q(s,a)**: "HOW Good Is This Specific Action?" (then Following Policy)the V-q Relationship Is like Asking:- V: "HOW Well Will I Do from This Chess Position?"- Q: "HOW Well Will I Do If I Move My Queen Here, Then Play Normally?"](
#part-3-action-value-functions-q-functions-exercise-31-computing-q-valuesdefinitionthe-action-value-function-qπsa-represents-the-expected-return-when-taking-action-a-in-state-s-and-then-following-policy-πkey-question-q-functions-answerq-functions-answer-what-if-i-take-this-specific-action-here-then-follow-my-policymathematical-relationshipsv-from-q-policy-weighted-averagevπs--suma-πas-qπsaq-from-v-bellman-backupqπsa--sums-pssarsas--γvπsbellman-equation-for-qqπsa--sums-pssarsas--γ-suma-πasqπsaintuition--vs-how-good-is-this-state-following-current-policy--qsa-how-good-is-this-specific-action-then-following-policythe-v-q-relationship-is-like-asking--v-how-well-will-i-do-from-this-chess-position--q-how-well-will-i-do-if-i-move-my-queen-here-then-play-normally)
- [🎯 Q-functions Deep Dive: the "what If" Values**core Concept:**q-functions Provide Action-specific Evaluations, Allowing Us to Compare Different Choices Directly.---
#
#
# 🍕 Restaurant Decision Analogy**scenario:** You're Choosing a Restaurant from Downtown Location.**value Functions:**- **v(downtown)** = 7.5 → "average Satisfaction from This Location with My Usual Choices"- **q(downtown, Pizza*place)** = 8.2 → "satisfaction If I Specifically Choose Pizza"- **q(downtown, Sushi*place)** = 6.8 → "satisfaction If I Specifically Choose Sushi"- **q(downtown, Burger*place)** = 7.1 → "satisfaction If I Specifically Choose Burgers"**policy Calculation:**if My Policy Is 50% Pizza, 30% Sushi, 20% Burgers:v(downtown) = 0.5×8.2 + 0.3×6.8 + 0.2×7.1 = 4.1 + 2.04 + 1.42 = 7.56 ✓---
#
#
# 🧮 Mathematical Relationships EXPLAINED**1. V from Q (weighted Average):**$$v^π(s) = \sum*a Π(a|s) × Q^π(s,a)$$**interpretation:** State Value = Probability of Each Action × Value of That ACTION**2. Q from V (bellman Backup):**$$q^π(s,a) = \sum*{s'} P(s'|s,a) × [r(s,a,s') + Γv^π(s')]$$**interpretation:** Action Value = Immediate Reward + Discounted Future State Value---
#
#
# 🔥 Why Q-functions MATTER**1. Direct Action Comparison:**- Q(s, Left) = 5.2 Vs Q(s, Right) = 7.8 → Choose Right!- No Need to Compute State Values FIRST**2. Policy Improvement:**- Π*new(s) = Argmax*a Q^π*old(s,a)- Directly Find the Best ACTION**3. Optimal Decision Making:**- Q*(s,a) Tells Us the Value of Each Action under Optimal Behavior- Essential for Q-learning Algorithms---
#
#
# 📊 Visual Understandingthink of Q-values as Action-specific "heat Maps":- **hot Spots** (high Q-values): Good Actions to Take- **cold Spots** (LOW Q-values): Actions to Avoid- **separate Map for Each Action**: Q(s,↑), Q(s,↓), Q(s,←), Q(s,→)**gridworld Example:**- Q(state, "toward*goal") Typically Has Higher Values- Q(state, "toward*obstacle") Typically Has Lower Values- Q(state, "toward*wall") Often Has Negative Values - Like: "restaurant Satisfaction = Meal Quality + How I'll Feel Tomorrow"
#
#
#
# **why Q-functions MATTER**1. **better Decision Making**: Q-values Directly Tell Us Which Action Is Best- Max*a Q(s,a) Gives the Best Action in State S2. **policy Improvement**: We Can Improve Policies by Being Greedy W.r.t. Q-values- Π*new(s) = Argmax*a Q^Π_OLD(S,A)3. **action Comparison**: Compare Different Actions in the Same State- "should I Go Left or Right from Here?"
#
#
#
# **visual Understanding**think of Q-values as a "heat Map" for Each Action:- **hot Spots** (high Q-values): Good Actions to Take- **cold Spots** (LOW Q-values): Actions to Avoid - **different Maps for Each Action**: Q(s,up), Q(s,down), Q(s,left), Q(s,right)
#
#
#
# **common Confusion: V Vs Q**- **v(s)**: "HOW Good Is My Current Strategy from This Position?"- **q(s,a)**: "HOW Good Is This Specific Move, Then Using My Strategy?"it's like Asking:- V: "HOW Well Will I Do in This Chess Position?" - Q: "HOW Well Will I Do If I Move My Queen Here, Then Play Normally?"](
#-q-functions-deep-dive-the-what-if-valuescore-conceptq-functions-provide-action-specific-evaluations-allowing-us-to-compare-different-choices-directly-----restaurant-decision-analogyscenario-youre-choosing-a-restaurant-from-downtown-locationvalue-functions--vdowntown--75--average-satisfaction-from-this-location-with-my-usual-choices--qdowntown-pizzaplace--82--satisfaction-if-i-specifically-choose-pizza--qdowntown-sushiplace--68--satisfaction-if-i-specifically-choose-sushi--qdowntown-burgerplace--71--satisfaction-if-i-specifically-choose-burgerspolicy-calculationif-my-policy-is-50-pizza-30-sushi-20-burgersvdowntown--0582--0368--0271--41--204--142--756------mathematical-relationships-explained1-v-from-q-weighted-averagevπs--suma-πas--qπsainterpretation-state-value--probability-of-each-action--value-of-that-action2-q-from-v-bellman-backupqπsa--sums-pssa--rsas--γvπsinterpretation-action-value--immediate-reward--discounted-future-state-value-----why-q-functions-matter1-direct-action-comparison--qs-left--52-vs-qs-right--78--choose-right--no-need-to-compute-state-values-first2-policy-improvement--πnews--argmaxa-qπoldsa--directly-find-the-best-action3-optimal-decision-making--qsa-tells-us-the-value-of-each-action-under-optimal-behavior--essential-for-q-learning-algorithms-----visual-understandingthink-of-q-values-as-action-specific-heat-maps--hot-spots-high-q-values-good-actions-to-take--cold-spots-low-q-values-actions-to-avoid--separate-map-for-each-action-qs-qs-qs-qsgridworld-example--qstate-towardgoal-typically-has-higher-values--qstate-towardobstacle-typically-has-lower-values--qstate-towardwall-often-has-negative-values---like-restaurant-satisfaction--meal-quality--how-ill-feel-tomorrow-why-q-functions-matter1-better-decision-making-q-values-directly-tell-us-which-action-is-best--maxa-qsa-gives-the-best-action-in-state-s2-policy-improvement-we-can-improve-policies-by-being-greedy-wrt-q-values--πnews--argmaxa-qπ_oldsa3-action-comparison-compare-different-actions-in-the-same-state--should-i-go-left-or-right-from-here-visual-understandingthink-of-q-values-as-a-heat-map-for-each-action--hot-spots-high-q-values-good-actions-to-take--cold-spots-low-q-values-actions-to-avoid---different-maps-for-each-action-qsup-qsdown-qsleft-qsright-common-confusion-v-vs-q--vs-how-good-is-my-current-strategy-from-this-position--qsa-how-good-is-this-specific-move-then-using-my-strategyits-like-asking--v-how-well-will-i-do-in-this-chess-position---q-how-well-will-i-do-if-i-move-my-queen-here-then-play-normally)
- [Part 4: Policy Improvement and Policy Iteration
#
#
# Exercise 4.1: Policy Improvement**definition:**given a Value Function V^π, We Can Improve the Policy by Being Greedy with Respect to the Action-value Function.**policy Improvement Formula:**$$π'(s) = \arg\max*a Q^π(s,a) = \arg\max*a \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**interpretation:** Choose the Action That Maximizes Expected Return from Each State.**policy Improvement Theorem:**if Π' Is Greedy with Respect to V^π, Then V^π'(s) ≥ V^π(s) for All States S.**translation:** "IF I Always Choose the Best Action Based on My Current Understanding, I Can Only Do Better (OR at Least as Well)."---
#
#
# Exercise 4.2: Policy Iteration Algorithm**policy Iteration STEPS:**1. **initialize**: Start with Arbitrary Policy Π₀2. **repeat until Convergence**:- **policy Evaluation**: Compute V^π*k (solve Bellman Equation)- **policy Improvement**: Π*{K+1}(S) = Argmax*a Q^Π_K(S,A)3. **output**: Optimal Policy Π* and Value Function V***convergence Guarantee:** Policy Iteration Is Guaranteed to Converge to the Optimal Policy in Finite Time for Finite Mdps.**why It Works:**- Each Step Produces a Better (OR Equal) Policy- There Are Only Finitely Many Deterministic Policies- Must Eventually Reach Optimal Policy](
#part-4-policy-improvement-and-policy-iteration-exercise-41-policy-improvementdefinitiongiven-a-value-function-vπ-we-can-improve-the-policy-by-being-greedy-with-respect-to-the-action-value-functionpolicy-improvement-formulaπs--argmaxa-qπsa--argmaxa-sums-pssarsas--γvπsinterpretation-choose-the-action-that-maximizes-expected-return-from-each-statepolicy-improvement-theoremif-π-is-greedy-with-respect-to-vπ-then-vπs--vπs-for-all-states-stranslation-if-i-always-choose-the-best-action-based-on-my-current-understanding-i-can-only-do-better-or-at-least-as-well----exercise-42-policy-iteration-algorithmpolicy-iteration-steps1-initialize-start-with-arbitrary-policy-π₀2-repeat-until-convergence--policy-evaluation-compute-vπk-solve-bellman-equation--policy-improvement-πk1s--argmaxa-qπ_ksa3-output-optimal-policy-π-and-value-function-vconvergence-guarantee-policy-iteration-is-guaranteed-to-converge-to-the-optimal-policy-in-finite-time-for-finite-mdpswhy-it-works--each-step-produces-a-better-or-equal-policy--there-are-only-finitely-many-deterministic-policies--must-eventually-reach-optimal-policy)
- [🚀 Policy Improvement Deep Dive: Making Better Decisions**core Idea:** Use the Value Function to Make Better Action Choices.---
#
#
# 📚 Learning Process Analogy**scenario:** You're Learning to Play Chess.**policy Evaluation:** "HOW Good Is My Current Playing Style?"- Analyze Your Current Strategy- Evaluate Typical Game Outcomes- Identify Strengths and Weaknesses**policy Improvement:** "HOW Can I Play Better?"- Look at Each Position Where You Made Suboptimal Moves- Replace Bad Moves with Better Alternatives- Update Your Playing Strategy**policy Iteration:** Repeat This Cycle until You Can't Improve Further.---
#
#
# 🧮 Mathematical Foundations**policy Improvement Theorem:**if Π' Is Greedy W.r.t. V^π, Then V^π'(s) ≥ V^π(s) for All S.**proof INTUITION:**1. **greedy Action**: Choose a Such That Q^π(s,a) Is MAXIMIZED2. **definition**: Q^π(s,a) ≥ V^π(s) for the Chosen ACTION3. **new Policy**: Π'(s) Gives This Optimal ACTION4. **result**: V^π'(s) ≥ V^π(s)**why Greedy Improvement Works:**- Current Policy Chooses Actions with Average Value V^π(s)- Greedy Policy Chooses Action with Maximum Value Q^π(s,a)- Maximum ≥ Average, So New Policy Is Better---
#
#
# 🔄 Policy Iteration: the Complete Cycle**step 1 - Policy Evaluation:** "HOW Good Is My Current Policy?"```v^π(s) ← Expected Return Following Π from State S```**step 2 - Policy Improvement:** "what's the Best Action in Each State?"```π'(s) ← Action That Maximizes Q^π(s,a)```**step 3 - Check Convergence:** "DID My Policy Change?"```if Π'(s) = Π(s) for All S: Stop (optimal Found)else: Π ← Π' and Repeat```---
#
#
# 🎯 Key PROPERTIES**1. Monotonic IMPROVEMENT:**V^Π₀ ≤ V^Π₁ ≤ V^Π₂ ≤ ... ≤ V^Π**2. Finite Convergence:**algorithm Terminates in Finite Steps (FOR Finite MDPS)**3. Optimal Solution:**final Policy Π* Is Optimal: V^π* = V**4. Model-based:**requires Knowledge of Transition Probabilities P(s'|s,a) and Rewards R(s,a,s')think of a Student Improving Their Study STRATEGY:1. **current Strategy** (policy Π): "I Study Randomly for 2 HOURS"2. **evaluate Strategy** (policy Evaluation): "HOW Well Does This Work for Each Subject?" 3. **find Better Strategy** (policy Improvement): "math Needs 3 Hours, History Needs 1 HOUR"4. **repeat**: Keep Refining until No More Improvements Possible
#
#
#
# **mathematical Intuition**policy Improvement Theorem**: If Q^π(s,a) > V^π(s) for Some Action A, Then Taking Action a Is Better Than Following Policy Π.**translation**: "IF Doing Action a Gives Higher Value Than My Current Average, I Should Do Action a More Often!"**greedy Improvement**:```pythonπ*new(s) = Argmax*a Q^π(s,a)```"always Choose the Action with Highest Q-value"
#
#
#
# **why Does This Work?**monotonic Improvement**: Each Policy Improvement Step Makes the Policy at Least as Good, Usually Better.**proof Sketch**:- If We're Greedy W.r.t. Q^π, We Get V^π_new ≥ V^π- "IF I Always Choose the Best Available Action, I Can't Do Worse"
#
#
#
# **policy Iteration: the Complete Algorithm**the Cycle**:```random Policy → Evaluate → Improve → Evaluate → Improve → ... → Optimal Policy```**why It CONVERGES**:1. **finite State/action Space**: Limited Number of Possible POLICIES2. **monotonic Improvement**: Each Step Makes Policy Better (OR SAME)3. **NO Cycles**: Can't Go Backwards to a Worse POLICY4. **must Terminate**: Eventually Reach Optimal Policy
#
#
#
# **real-world Example: Learning to Drive**iteration 1**:- **policy**: "drive Slowly Everywhere" - **evaluation**: "safe but Inefficient on Highways"- **improvement**: "drive Fast on Highways, Slow in Neighborhoods"**iteration 2**:- **policy**: "speed Varies by Road Type"- **evaluation**: "good, but Inefficient in Traffic" - **improvement**: "also Consider Traffic Conditions"**final Policy**: "optimal Speed Based on Road Type, Traffic, Weather, Etc."
#
#
#
# **key INSIGHTS**1. **guaranteed Improvement**: Policy Iteration Always Finds the Optimal Policy (FOR Finite MDPS)2. **fast Convergence**: Usually Converges in Just a Few ITERATIONS3. **NO Exploration Needed**: Uses Complete Model Knowledge (unlike Q-learning LATER)4. **computational Cost**: Each Iteration Requires Solving the Bellman Equation
#
#
#
# **common Pitfalls**- **getting Stuck**: in Stochastic Environments, Might Need Exploration- **computational Cost**: Policy Evaluation Can Be Expensive - **model Required**: Need to Know P(s'|s,a) and R(s,a,s')](
#-policy-improvement-deep-dive-making-better-decisionscore-idea-use-the-value-function-to-make-better-action-choices-----learning-process-analogyscenario-youre-learning-to-play-chesspolicy-evaluation-how-good-is-my-current-playing-style--analyze-your-current-strategy--evaluate-typical-game-outcomes--identify-strengths-and-weaknessespolicy-improvement-how-can-i-play-better--look-at-each-position-where-you-made-suboptimal-moves--replace-bad-moves-with-better-alternatives--update-your-playing-strategypolicy-iteration-repeat-this-cycle-until-you-cant-improve-further-----mathematical-foundationspolicy-improvement-theoremif-π-is-greedy-wrt-vπ-then-vπs--vπs-for-all-sproof-intuition1-greedy-action-choose-a-such-that-qπsa-is-maximized2-definition-qπsa--vπs-for-the-chosen-action3-new-policy-πs-gives-this-optimal-action4-result-vπs--vπswhy-greedy-improvement-works--current-policy-chooses-actions-with-average-value-vπs--greedy-policy-chooses-action-with-maximum-value-qπsa--maximum--average-so-new-policy-is-better-----policy-iteration-the-complete-cyclestep-1---policy-evaluation-how-good-is-my-current-policyvπs--expected-return-following-π-from-state-sstep-2---policy-improvement-whats-the-best-action-in-each-stateπs--action-that-maximizes-qπsastep-3---check-convergence-did-my-policy-changeif-πs--πs-for-all-s-stop-optimal-foundelse-π--π-and-repeat-----key-properties1-monotonic-improvementvπ₀--vπ₁--vπ₂----vπ2-finite-convergencealgorithm-terminates-in-finite-steps-for-finite-mdps3-optimal-solutionfinal-policy-π-is-optimal-vπ--v4-model-basedrequires-knowledge-of-transition-probabilities-pssa-and-rewards-rsasthink-of-a-student-improving-their-study-strategy1-current-strategy-policy-π-i-study-randomly-for-2-hours2-evaluate-strategy-policy-evaluation-how-well-does-this-work-for-each-subject-3-find-better-strategy-policy-improvement-math-needs-3-hours-history-needs-1-hour4-repeat-keep-refining-until-no-more-improvements-possible-mathematical-intuitionpolicy-improvement-theorem-if-qπsa--vπs-for-some-action-a-then-taking-action-a-is-better-than-following-policy-πtranslation-if-doing-action-a-gives-higher-value-than-my-current-average-i-should-do-action-a-more-oftengreedy-improvementpythonπnews--argmaxa-qπsaalways-choose-the-action-with-highest-q-value-why-does-this-workmonotonic-improvement-each-policy-improvement-step-makes-the-policy-at-least-as-good-usually-betterproof-sketch--if-were-greedy-wrt-qπ-we-get-vπ_new--vπ--if-i-always-choose-the-best-available-action-i-cant-do-worse-policy-iteration-the-complete-algorithmthe-cyclerandom-policy--evaluate--improve--evaluate--improve----optimal-policywhy-it-converges1-finite-stateaction-space-limited-number-of-possible-policies2-monotonic-improvement-each-step-makes-policy-better-or-same3-no-cycles-cant-go-backwards-to-a-worse-policy4-must-terminate-eventually-reach-optimal-policy-real-world-example-learning-to-driveiteration-1--policy-drive-slowly-everywhere---evaluation-safe-but-inefficient-on-highways--improvement-drive-fast-on-highways-slow-in-neighborhoodsiteration-2--policy-speed-varies-by-road-type--evaluation-good-but-inefficient-in-traffic---improvement-also-consider-traffic-conditionsfinal-policy-optimal-speed-based-on-road-type-traffic-weather-etc-key-insights1-guaranteed-improvement-policy-iteration-always-finds-the-optimal-policy-for-finite-mdps2-fast-convergence-usually-converges-in-just-a-few-iterations3-no-exploration-needed-uses-complete-model-knowledge-unlike-q-learning-later4-computational-cost-each-iteration-requires-solving-the-bellman-equation-common-pitfalls--getting-stuck-in-stochastic-environments-might-need-exploration--computational-cost-policy-evaluation-can-be-expensive---model-required-need-to-know-pssa-and-rsas)
- [Part 5: Experiments and Analysis
#
#
# Exercise 5.1: Effect of Discount Factor (γ)**definition:**the Discount Factor Γ Determines How Much We Value Future Rewards Compared to Immediate Rewards.**mathematical Impact:**$$g*t = R*{T+1} + ΓR*{T+2} + Γ^2R*{T+3} + ... = \SUM*{K=0}^{\INFTY} Γ^k R*{T+K+1}$$**INTERPRETATION of Different Values:**- **Γ = 0**: Only Immediate Rewards Matter (myopic Behavior)- **Γ = 1**: All Future Rewards Equally Important (infinite Horizon)- **0 < Γ < 1**: Future Rewards Are Discounted (realistic)**task:** Experiment with Different Discount Factors and Analyze Their Effect on the Optimal Policy.**research QUESTIONS:**1. How Does Γ Affect the Optimal POLICY?2. Which Γ Values Lead to Faster CONVERGENCE?3. What Happens to State Values as Γ Changes?](
#part-5-experiments-and-analysis-exercise-51-effect-of-discount-factor-γdefinitionthe-discount-factor-γ-determines-how-much-we-value-future-rewards-compared-to-immediate-rewardsmathematical-impactgt--rt1--γrt2--γ2rt3----sumk0infty-γk-rtk1interpretation-of-different-values--γ--0-only-immediate-rewards-matter-myopic-behavior--γ--1-all-future-rewards-equally-important-infinite-horizon--0--γ--1-future-rewards-are-discounted-realistictask-experiment-with-different-discount-factors-and-analyze-their-effect-on-the-optimal-policyresearch-questions1-how-does-γ-affect-the-optimal-policy2-which-γ-values-lead-to-faster-convergence3-what-happens-to-state-values-as-γ-changes)
- [💰 Discount Factor Deep Dive: Balancing Present Vs Future**core Concept:** the Discount Factor Γ Controls the Agent's "patience" or Time Preference.---
#
#
# ⏰ Time Value of Rewards**financial Analogy:**just like Money, Rewards Have "time Value":- $100 Today Vs $100 in 10 Years → Most Prefer Today (inflation, Uncertainty)- +10 Reward Now Vs +10 Reward in 100 Time Steps → Usually Prefer Immediate**mathematical Effect:**- **Γ = 0.1**: Reward 10 Steps Away Is Worth 0.1¹⁰ = 0.0000000001 of Current Reward- **Γ = 0.9**: Reward 10 Steps Away Is Worth 0.9¹⁰ = 0.35 of Current Reward- **Γ = 0.99**: Reward 10 Steps Away Is Worth 0.99¹⁰ = 0.90 of Current Reward---
#
#
# 🌎 Real-world Analogies**γ = 0.1 (very Impatient/myopic):**- 🍕 "I Want Pizza Now, Don't Care About Health Consequences"- 💳 "BUY with Credit Card, Ignore Interest Charges"- 🚗 "take Fastest Route, Ignore Traffic Fines"**γ = 0.5 (moderately Patient):**- 🏃 "exercise Sometimes for Health Benefits"- 💰 "save Some Money, Spend Some Now"- 📚 "study When Motivated, Party When Not"**γ = 0.9 (balanced):**- 💪 "exercise Regularly for Long-term Health"- 🎓 "study Hard Now for Career Benefits Later"- 💰 "invest Consistently for Retirement"**γ = 0.99 (very Patient):**- 🌱 "plant Trees for Future Generations"- 🏠 "BUY House as Long-term Investment"- 🌍 "address Climate Change for Distant Future"---
#
#
# 📊 Effect on Optimal Policy**low Γ (myopic Behavior):**- Takes Shortest Immediate Path to Reward- Ignores Long-term Consequences- May Get Stuck in Local Optima- Fast Convergence but Potentially Poor Solutions**high Γ (farsighted Behavior):**- Considers Long-term Consequences- May Take Longer Paths for Better Future Outcomes- Explores More Thoroughly- Slower Convergence but Better Final Solutions**in Gridworld Context:**- **low Γ**: Rushes toward Goal, Ignoring Obstacles- **high Γ**: Carefully Plans Path, Avoids Risky Moves
#
#
#
# **mathematical Impact**return Formula**: G*t = R*{T+1} + ΓR*{T+2} + Γ²R*{T+3} + Γ³R*{T+4} + ...**examples**:**γ = 0.9** (patient Agent):- G*t = R*{T+1} + 0.9×R*{T+2} + 0.81×R*{T+3} + 0.729×R*{T+4} + ...- Reward in 1 Step: Worth 100% of Immediate Reward- Reward in 2 Steps: Worth 90% of Immediate Reward - Reward in 3 Steps: Worth 81% of Immediate Reward- Reward in 10 Steps: Worth 35% of Immediate Reward**γ = 0.1** (impatient Agent):- G*t = R*{T+1} + 0.1×R*{T+2} + 0.01×R*{T+3} + 0.001×R_{T+4} + ...- Reward in 2 Steps: Worth Only 10% of Immediate Reward- Reward in 3 Steps: Worth Only 1% of Immediate Reward- Very Myopic - Only Cares About Next Few Steps
#
#
#
# **real-world Analogies**γ = 0.1** (very Impatient):- 🍕 "I Want Pizza Now, Don't Care About Health Consequences"- 📱 "BUY the Cheapest Phone, Ignore Long-term Durability" - 🚗 "take the Fastest Route, Ignore Traffic Fines"**γ = 0.9** (balanced):- 💪 "exercise Now for Health Benefits Later"- 🎓 "study Hard Now for Career Benefits Later"- 💰 "invest Money for Retirement"**γ = 0.99** (very Patient):- 🌱 "plant Trees for Future Generations"- 🏠 "BUY a House as Long-term Investment"- 🌍 "address Climate Change for Distant Future"
#
#
#
# **effect on Optimal Policy**low Γ (myopic Behavior)**:- Takes Shortest Path to Goal- Ignores Long-term Consequences - Might Take Dangerous Shortcuts- Policy: "rush to Goal, Avoid Obstacles Minimally"**high Γ (farsighted Behavior)**:- Takes Safer, Longer Paths- Values Long-term Safety- More Conservative Decisions- Policy: "GET to Goal Safely, Even If It Takes Longer"
#
#
#
# **choosing Γ in PRACTICE**CONSIDER**:1. **problem Horizon**: Short-term Tasks → Lower Γ, Long-term Tasks → Higher Γ2. **uncertainty**: More Uncertain Future → Lower Γ3. **safety**: Safety-critical Applications → Higher Γ4. **computational**: Higher Γ → Slower Convergence**common Values**:- **Γ = 0.9**: General Purpose, Good Balance- **Γ = 0.95-0.99**: Long-term Planning Tasks- **Γ = 0.1-0.5**: Short-term Reactive Tasks- **Γ = 1.0**: Infinite Horizon, Theoretical Studies (CAN Cause Issues)
#
#
#
# **debugging with Γ**if Your Agent:- **ignores Long-term Rewards**: Increase Γ- **IS Too Conservative**: Decrease Γ - **won't Converge**: Check If Γ Is Too Close to 1- **makes Random Decisions**: Γ Might Be Too Low](
#-discount-factor-deep-dive-balancing-present-vs-futurecore-concept-the-discount-factor-γ-controls-the-agents-patience-or-time-preference-----time-value-of-rewardsfinancial-analogyjust-like-money-rewards-have-time-value--100-today-vs-100-in-10-years--most-prefer-today-inflation-uncertainty--10-reward-now-vs-10-reward-in-100-time-steps--usually-prefer-immediatemathematical-effect--γ--01-reward-10-steps-away-is-worth-01¹⁰--00000000001-of-current-reward--γ--09-reward-10-steps-away-is-worth-09¹⁰--035-of-current-reward--γ--099-reward-10-steps-away-is-worth-099¹⁰--090-of-current-reward-----real-world-analogiesγ--01-very-impatientmyopic---i-want-pizza-now-dont-care-about-health-consequences---buy-with-credit-card-ignore-interest-charges---take-fastest-route-ignore-traffic-finesγ--05-moderately-patient---exercise-sometimes-for-health-benefits---save-some-money-spend-some-now---study-when-motivated-party-when-notγ--09-balanced---exercise-regularly-for-long-term-health---study-hard-now-for-career-benefits-later---invest-consistently-for-retirementγ--099-very-patient---plant-trees-for-future-generations---buy-house-as-long-term-investment---address-climate-change-for-distant-future-----effect-on-optimal-policylow-γ-myopic-behavior--takes-shortest-immediate-path-to-reward--ignores-long-term-consequences--may-get-stuck-in-local-optima--fast-convergence-but-potentially-poor-solutionshigh-γ-farsighted-behavior--considers-long-term-consequences--may-take-longer-paths-for-better-future-outcomes--explores-more-thoroughly--slower-convergence-but-better-final-solutionsin-gridworld-context--low-γ-rushes-toward-goal-ignoring-obstacles--high-γ-carefully-plans-path-avoids-risky-moves-mathematical-impactreturn-formula-gt--rt1--γrt2--γ²rt3--γ³rt4--examplesγ--09-patient-agent--gt--rt1--09rt2--081rt3--0729rt4----reward-in-1-step-worth-100-of-immediate-reward--reward-in-2-steps-worth-90-of-immediate-reward---reward-in-3-steps-worth-81-of-immediate-reward--reward-in-10-steps-worth-35-of-immediate-rewardγ--01-impatient-agent--gt--rt1--01rt2--001rt3--0001r_t4----reward-in-2-steps-worth-only-10-of-immediate-reward--reward-in-3-steps-worth-only-1-of-immediate-reward--very-myopic---only-cares-about-next-few-steps-real-world-analogiesγ--01-very-impatient---i-want-pizza-now-dont-care-about-health-consequences---buy-the-cheapest-phone-ignore-long-term-durability----take-the-fastest-route-ignore-traffic-finesγ--09-balanced---exercise-now-for-health-benefits-later---study-hard-now-for-career-benefits-later---invest-money-for-retirementγ--099-very-patient---plant-trees-for-future-generations---buy-a-house-as-long-term-investment---address-climate-change-for-distant-future-effect-on-optimal-policylow-γ-myopic-behavior--takes-shortest-path-to-goal--ignores-long-term-consequences---might-take-dangerous-shortcuts--policy-rush-to-goal-avoid-obstacles-minimallyhigh-γ-farsighted-behavior--takes-safer-longer-paths--values-long-term-safety--more-conservative-decisions--policy-get-to-goal-safely-even-if-it-takes-longer-choosing-γ-in-practiceconsider1-problem-horizon-short-term-tasks--lower-γ-long-term-tasks--higher-γ2-uncertainty-more-uncertain-future--lower-γ3-safety-safety-critical-applications--higher-γ4-computational-higher-γ--slower-convergencecommon-values--γ--09-general-purpose-good-balance--γ--095-099-long-term-planning-tasks--γ--01-05-short-term-reactive-tasks--γ--10-infinite-horizon-theoretical-studies-can-cause-issues-debugging-with-γif-your-agent--ignores-long-term-rewards-increase-γ--is-too-conservative-decrease-γ---wont-converge-check-if-γ-is-too-close-to-1--makes-random-decisions-γ-might-be-too-low)
- [Exercise 5.2: Modified Environment Experiments**task A**: Modify the Reward Structure and Analyze How It Affects the Optimal Policy:- Change Step Reward from -0.1 to -1.0 (higher Cost for Each Step)- Change Goal Reward from 10 to 5- Add Positive Rewards for Certain States**task B**: Experiment with Different Obstacle Configurations:- Remove Some Obstacles- Add More Obstacles- Change Obstacle Positions**task C**: Test with Different Starting Positions and Analyze Convergence.](
#exercise-52-modified-environment-experimentstask-a-modify-the-reward-structure-and-analyze-how-it-affects-the-optimal-policy--change-step-reward-from--01-to--10-higher-cost-for-each-step--change-goal-reward-from-10-to-5--add-positive-rewards-for-certain-statestask-b-experiment-with-different-obstacle-configurations--remove-some-obstacles--add-more-obstacles--change-obstacle-positionstask-c-test-with-different-starting-positions-and-analyze-convergence)
- [Part 6: Summary and Key Takeaways
#
#
# What We've LEARNED**1. Markov Decision Processes (mdps):**- **framework**: Sequential Decision Making under Uncertainty- **components**: (S, A, P, R, Γ) - States, Actions, Transitions, Rewards, Discount- **markov Property**: Future Depends Only on Current State, Not History- **foundation**: Mathematical Basis for All Rl ALGORITHMS**2. Value Functions:**- **v^π(s)**: Expected Return Starting from State S Following Policy Π - **q^π(s,a)**: Expected Return Taking Action a in State S, Then Following Π- **relationship**: V^π(s) = Σ*a Π(a|s) Q^π(s,a)- **purpose**: Measure "goodness" of States and ACTIONS**3. Bellman Equations:**- **for V**: V^π(s) = Σ*a Π(a|s) Σ*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]- **for Q**: Q^π(s,a) = Σ*{s'} P(s'|s,a)[r(s,a,s') + Γ Σ*{a'} Π(a'|s')q^π(s',a')]- **significance**: Recursive Relationship Enabling Dynamic Programming SOLUTIONS**4. Policy Evaluation:**- **algorithm**: Iterative Method to Compute V^π Given Policy Π- **convergence**: Guaranteed for Finite Mdps with Γ < 1- **application**: Foundation for Policy Iteration and Value ITERATION**5. Policy Improvement:**- **theorem**: Greedy Policy W.r.t. V^π Is at Least as Good as Π- **formula**: Π'(s) = Argmax*a Q^π(s,a)- **monotonicity**: Each Improvement Step Yields Better or Equal POLICY**6. Policy Iteration:**- **algorithm**: Alternates between Evaluation and Improvement- **guarantee**: Converges to Optimal Policy Π*- **efficiency**: Usually Converges in Few Iterations---
#
#
# Key Insights from Experiments**discount Factor (Γ) Effects:**- **low Γ**: Myopic Behavior, Focuses on Immediate Rewards- **high Γ**: Farsighted Behavior, Considers Long-term Consequences- **trade-off**: Convergence Speed Vs Solution Quality**environment Structure Impact:**- **reward Structure**: Significantly Affects Optimal Policy- **obstacles**: Create Navigation Challenges Requiring Planning- **starting Position**: Can Influence Learning Dynamics**algorithm Characteristics:**- **model-based**: Requires Knowledge of P(s'|s,a) and R(s,a,s')- **exact Solution**: Finds Truly Optimal Policy (unlike Approximate Methods)- **computational Cost**: Scales with State Space Size---
#
#
# Connections to Advanced Topics**what This Enables:**- **value Iteration**: Direct Optimization of Value Function- **q-learning**: Model-free Learning of Action-value Functions- **deep Rl**: Neural Network Function Approximation- **policy Gradients**: Direct Policy Optimization Methods**next Steps in Learning:**- **temporal Difference Learning**: Learn from Incomplete Episodes- **function Approximation**: Handle Large/continuous State Spaces- **exploration Vs Exploitation**: Balance Learning and Performance- **multi-agent Systems**: Multiple Learning Agents Interacting---
#
#
# Reflection Questions**theoretical UNDERSTANDING:**1. How Would Stochastic Transitions Affect the Optimal POLICY?2. What Happens with Continuous State or Action SPACES?3. How Do We Handle Unknown Environment DYNAMICS?4. What Are Computational Limits for Large State Spaces?**practical APPLICATIONS:**1. How Could You Apply Mdps to Real-world Decision PROBLEMS?2. What Modifications Would Be Needed for Competitive SCENARIOS?3. How Would You Handle Partially Observable ENVIRONMENTS?4. What Safety Considerations Are Important in Rl Applications?](
#part-6-summary-and-key-takeaways-what-weve-learned1-markov-decision-processes-mdps--framework-sequential-decision-making-under-uncertainty--components-s-a-p-r-γ---states-actions-transitions-rewards-discount--markov-property-future-depends-only-on-current-state-not-history--foundation-mathematical-basis-for-all-rl-algorithms2-value-functions--vπs-expected-return-starting-from-state-s-following-policy-π---qπsa-expected-return-taking-action-a-in-state-s-then-following-π--relationship-vπs--σa-πas-qπsa--purpose-measure-goodness-of-states-and-actions3-bellman-equations--for-v-vπs--σa-πas-σs-pssarsas--γvπs--for-q-qπsa--σs-pssarsas--γ-σa-πasqπsa--significance-recursive-relationship-enabling-dynamic-programming-solutions4-policy-evaluation--algorithm-iterative-method-to-compute-vπ-given-policy-π--convergence-guaranteed-for-finite-mdps-with-γ--1--application-foundation-for-policy-iteration-and-value-iteration5-policy-improvement--theorem-greedy-policy-wrt-vπ-is-at-least-as-good-as-π--formula-πs--argmaxa-qπsa--monotonicity-each-improvement-step-yields-better-or-equal-policy6-policy-iteration--algorithm-alternates-between-evaluation-and-improvement--guarantee-converges-to-optimal-policy-π--efficiency-usually-converges-in-few-iterations----key-insights-from-experimentsdiscount-factor-γ-effects--low-γ-myopic-behavior-focuses-on-immediate-rewards--high-γ-farsighted-behavior-considers-long-term-consequences--trade-off-convergence-speed-vs-solution-qualityenvironment-structure-impact--reward-structure-significantly-affects-optimal-policy--obstacles-create-navigation-challenges-requiring-planning--starting-position-can-influence-learning-dynamicsalgorithm-characteristics--model-based-requires-knowledge-of-pssa-and-rsas--exact-solution-finds-truly-optimal-policy-unlike-approximate-methods--computational-cost-scales-with-state-space-size----connections-to-advanced-topicswhat-this-enables--value-iteration-direct-optimization-of-value-function--q-learning-model-free-learning-of-action-value-functions--deep-rl-neural-network-function-approximation--policy-gradients-direct-policy-optimization-methodsnext-steps-in-learning--temporal-difference-learning-learn-from-incomplete-episodes--function-approximation-handle-largecontinuous-state-spaces--exploration-vs-exploitation-balance-learning-and-performance--multi-agent-systems-multiple-learning-agents-interacting----reflection-questionstheoretical-understanding1-how-would-stochastic-transitions-affect-the-optimal-policy2-what-happens-with-continuous-state-or-action-spaces3-how-do-we-handle-unknown-environment-dynamics4-what-are-computational-limits-for-large-state-spacespractical-applications1-how-could-you-apply-mdps-to-real-world-decision-problems2-what-modifications-would-be-needed-for-competitive-scenarios3-how-would-you-handle-partially-observable-environments4-what-safety-considerations-are-important-in-rl-applications)
- [🧠 Common Misconceptions and Intuitive Understandingbefore We Wrap Up, Let's Address Some Common Confusions and Solidify Understanding:
#
#
#
# **❌ Common MISCONCEPTIONS**1. "value Functions Are Just Rewards"**- ❌ Wrong: V(s) ≠ R(s) - ✅ Correct: V(s) = Expected Total Future Reward from State S- 🔍 Think: V(s) Is like Your Bank Account Balance, R(s) Is Your Daily INCOME**2. "q(s,a) Tells Me the Best Action"**- ❌ Wrong: Q(s,a) Is Not Binary Good/bad- ✅ Correct: Q(s,a) Is the Expected Value of Taking Action A- 🔍 Think: Compare Q-values to Choose Best Action: Argmax_a Q(S,A)**3. "policy Iteration Always Takes Many Steps"**- ❌ Wrong: Often Converges in 2-4 Iterations- ✅ Correct: Convergence Is Usually Very Fast- 🔍 Think: Once You Find a Good Strategy, Small Improvements Are ENOUGH**4. "random Policy Is Always Bad"**- ❌ Wrong: Random Policy Can Be Good for Exploration- ✅ Correct: Depends on Environment and Goals- 🔍 Think: Sometimes Trying New Things Leads to Better Discoveries
#
#
#
# **🎯 Key Intuitions to REMEMBER**1. the Big Picture Flow**:```environment → Policy → Actions → Rewards → Better Policy → REPEAT```**2. Value Functions as Gps**:- V(s): "HOW Good Is This Location Overall?"- Q(s,a): "HOW Good Is Taking This Road from This LOCATION?"**3. Bellman Equations as Consistency**:- "MY Value Should Equal Immediate Reward + Discounted Future Value"- Like: "MY Wealth = Today's Income + Tomorrow's WEALTH"**4. Policy Improvement as Learning**:- "IF I Know What Each Action Leads To, I Can Choose Better Actions"- Like: "IF I Know Exam Results for Each Study Method, I Can Study Better"
#
#
#
# **🔧 Troubleshooting Guide****if Values Don't Converge**:- Check If Γ < 1 - Reduce Convergence Threshold (theta)- Check for Bugs in Transition Probabilities**if Policy Doesn't Improve**:- Environment Might Be Too Simple (already Optimal)- Check Reward Structure - Might Need More Differentiation- Verify Policy Improvement Logic**if Results Seem Weird**:- Visualize Value Functions and Policies- Start with Simpler Environment- Check Reward Signs (positive/negative)
#
#
#
# **🚀 Connecting to Future Topics**what We Learned Here Enables:- **value Iteration**: Direct Value Optimization (next Week!)- **q-learning**: Learn Q-values without Knowing the Model- **deep Rl**: Use Neural Networks to Handle Large State Spaces- **policy Gradients**: Directly Optimize the Policy Parameters
#
#
#
# **🎭 the Rl Mindset**think like an Rl AGENT:1. **observe** Your Current Situation (STATE)2. **consider** Your Options (actions) 3. **predict** Outcomes (USE Your MODEL/EXPERIENCE)4. **choose** the Best Option (POLICY)5. **learn** from Results (update VALUES/POLICY)6. **repeat** until Masterythis Mindset Applies To:- Career Decisions- Investment Choices - Game Strategies- Daily Life Optimization](
#-common-misconceptions-and-intuitive-understandingbefore-we-wrap-up-lets-address-some-common-confusions-and-solidify-understanding--common-misconceptions1-value-functions-are-just-rewards---wrong-vs--rs----correct-vs--expected-total-future-reward-from-state-s---think-vs-is-like-your-bank-account-balance-rs-is-your-daily-income2-qsa-tells-me-the-best-action---wrong-qsa-is-not-binary-goodbad---correct-qsa-is-the-expected-value-of-taking-action-a---think-compare-q-values-to-choose-best-action-argmax_a-qsa3-policy-iteration-always-takes-many-steps---wrong-often-converges-in-2-4-iterations---correct-convergence-is-usually-very-fast---think-once-you-find-a-good-strategy-small-improvements-are-enough4-random-policy-is-always-bad---wrong-random-policy-can-be-good-for-exploration---correct-depends-on-environment-and-goals---think-sometimes-trying-new-things-leads-to-better-discoveries--key-intuitions-to-remember1-the-big-picture-flowenvironment--policy--actions--rewards--better-policy--repeat2-value-functions-as-gps--vs-how-good-is-this-location-overall--qsa-how-good-is-taking-this-road-from-this-location3-bellman-equations-as-consistency--my-value-should-equal-immediate-reward--discounted-future-value--like-my-wealth--todays-income--tomorrows-wealth4-policy-improvement-as-learning--if-i-know-what-each-action-leads-to-i-can-choose-better-actions--like-if-i-know-exam-results-for-each-study-method-i-can-study-better--troubleshooting-guideif-values-dont-converge--check-if-γ--1---reduce-convergence-threshold-theta--check-for-bugs-in-transition-probabilitiesif-policy-doesnt-improve--environment-might-be-too-simple-already-optimal--check-reward-structure---might-need-more-differentiation--verify-policy-improvement-logicif-results-seem-weird--visualize-value-functions-and-policies--start-with-simpler-environment--check-reward-signs-positivenegative--connecting-to-future-topicswhat-we-learned-here-enables--value-iteration-direct-value-optimization-next-week--q-learning-learn-q-values-without-knowing-the-model--deep-rl-use-neural-networks-to-handle-large-state-spaces--policy-gradients-directly-optimize-the-policy-parameters--the-rl-mindsetthink-like-an-rl-agent1-observe-your-current-situation-state2-consider-your-options-actions-3-predict-outcomes-use-your-modelexperience4-choose-the-best-option-policy5-learn-from-results-update-valuespolicy6-repeat-until-masterythis-mindset-applies-to--career-decisions--investment-choices---game-strategies--daily-life-optimization)



```python
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from collections import defaultdict
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Plotting configuration
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

print("Libraries imported successfully!")
print("NumPy version:", np.__version__)
```

    Libraries imported successfully!
    NumPy version: 2.3.1


#
# Part 1: Theoretical Foundation
#
#
# 1.1 Reinforcement Learning Framework**definition:**reinforcement Learning Is a Computational Approach to Learning from Interaction. the Key Elements Are:- **agent**: the Learner and Decision Maker - the Entity That Makes Choices- **environment**: the World the Agent Interacts with - Everything outside the Agent- **state (s)**: Current Situation of the Agent - Describes the Current Circumstances- **action (a)**: Choices Available to the Agent - Decisions That Can Be Made- **reward (r)**: Numerical Feedback from Environment - Immediate Feedback Signal- **policy (π)**: Agent's Strategy for Choosing Actions - Mapping from States to Actions**real-world Analogy:**think of Rl like Learning to Drive:- **agent** = the Driver (you)- **environment** = Roads, Traffic, Weather Conditions- **state** = Current Speed, Position, Traffic around You- **actions** = Accelerate, Brake, Turn Left/right- **reward** = Positive for Safe Driving, Negative for Accidents- **policy** = Your Driving Strategy (cautious, Aggressive, Etc.)---
#
#
# 1.2 Markov Decision Process (mdp)**definition:**an Mdp Is Defined by the Tuple (S, A, P, R, Γ) Where:- **s**: Set of States - All Possible Situations the Agent Can Encounter- **a**: Set of Actions - All Possible Decisions Available to the Agent- **p**: Transition Probability Function P(s'|s,a) - Probability of Moving to State S' Given Current State S and Action A- **r**: Reward Function R(s,a,s') - Immediate Reward Received for Transitioning from S to S' Via Action A- **γ**: Discount Factor (0 ≤ Γ ≤ 1) - Determines Importance of Future Rewards**markov Property:**the Future Depends Only on the Current State, Not on the History of How We Got There. MATHEMATICALLY:P(S*{T+1} = S' | S*t = S, A*t = A, S*{T-1}, A*{T-1}, ..., S*0, A*0) = P(S*{T+1} = S' | S*t = S, A*t = A)**intuition:**the Current State Contains All Information Needed to Make Optimal Decisions. the past Is Already "encoded" in the Current State.---
#
#
# 1.3 Value Functions**state-value Function:**$$v^π(s) = \mathbb{e}*π[g*t | S_t = S]$$**interpretation:** Expected Total Reward When Starting from State S and Following Policy Π. It Answers: "HOW Good Is It to Be in This State?"**action-value Function:**$$q^π(s,a) = \mathbb{e}*π[g*t | S*t = S, A*t = A]$$**interpretation:** Expected Total Reward When Taking Action a in State S and Then Following Policy Π. It Answers: "HOW Good Is It to Take This Specific Action in This State?"**return (cumulative Reward):**$$g*t = R*{T+1} + ΓR*{T+2} + Γ^2R*{T+3} + ... = \SUM*{K=0}^{\INFTY} Γ^k R*{T+K+1}$$**WHY Discount Factor Γ?**- **Γ = 0**: Only Immediate Rewards Matter (myopic)- **Γ = 1**: All Future Rewards Are Equally Important- **0 < Γ < 1**: Future Rewards Are Discounted (realistic for Most Scenarios)---
#
#
# 1.4 Bellman Equations**bellman Equation for State-value Function:**$$v^π(s) = \sum*a Π(a|s) \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**interpretation:** the Value of a State Equals the Immediate Reward Plus the Discounted Value of the Next State, Averaged over All Possible Actions and Transitions.**bellman Equation for Action-value Function:**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γ \sum*{a'} Π(a'|s')q^π(s',a')]$$**key Insight:** the Bellman Equations Express a Recursive Relationship - the Value of a State (OR State-action Pair) Depends on the Immediate Reward Plus the Discounted Value of Future States. This Is the Mathematical Foundation for Most Rl Algorithms.

#
#
# 📚 Common Misconceptions and Clarifications**misconception 1: "THE Agent Knows the Environment Model"**reality:** in Most Rl Problems, the Agent Doesn't Know P(s'|s,a) or R(s,a,s'). This Is Called "model-free" Rl, Where the Agent Learns through Trial and Error.**misconception 2: "higher Rewards Are Always Better"**reality:** the Goal Is to Maximize *cumulative* Reward, Not Immediate Reward. Sometimes Taking a Small Immediate Reward Prevents Getting a Much Larger Future Reward.**misconception 3: "THE Policy Should Always Be Deterministic"****reality:** Stochastic Policies (that Output Probabilities) Are Often Better Because They Allow for Exploration and Can Be Optimal in Certain Environments.---
#
#
# 🧠 Building Intuition: Restaurant Example**scenario:** You're Choosing Restaurants to Visit in a New City.**mdp Components:**- **states**: Your Hunger Level, Location, Time of Day, Budget- **actions**: Choose Restaurant A, B, C, or Cook at Home- **rewards**: Satisfaction from Food (immediate) + Health Effects (long-term)- **transitions**: How Your State Changes after Eating**value Functions:**- **v(hungry, Downtown, Evening)**: How Good Is This Situation Overall?- **q(hungry, Downtown, Evening, "restaurant A")**: How Good Is Choosing Restaurant a in This Situation?**policy Learning:** Initially Random Choices → Gradually Prefer Restaurants That Gave Good Experiences → Eventually Develop a Strategy That Considers Health, Taste, Cost, and Convenience.---
#
#
# 🔧 Mathematical Properties and Theorems**theorem 1: Existence and Uniqueness of Value Functions**for Any Policy Π and Finite Mdp, There Exists a Unique Solution to the Bellman Equations.**theorem 2: Bellman Optimality Principle**a Policy Π Is Optimal If and Only If:$$v^π(s) = \max_a Q^π(s,a) \text{ for All } S \IN S$$**theorem 3: Policy Improvement Theorem**if Π' Is Greedy with Respect to V^π, Then V^π'(s) ≥ V^π(s) for All States S.**practical Implications:**- We Can Always Improve a Policy by Being Greedy with Respect to Its Value Function- There Always Exists an Optimal Policy (MAY Not Be Unique)- the Optimal Value Function Satisfies the Bellman Optimality Equations- **rewards**: +1 for Safe Driving, -10 for Accidents, -1 for Speeding Tickets- **policy**: Your Driving Strategy (aggressive, Conservative, Etc.)
#
#
#
# **why Markov Property Matters**the **markov Property** Means "THE Future Depends Only on the Present, Not the Past."**example**: in Chess, to Decide Your Next Move, You Only Need to See the Current Board Position. You Don't Need to Know How the Pieces Got There - the Complete Game History Is Irrelevant for Making the Optimal Next Move.**non-markov Example**: Predicting Tomorrow's Weather Based Only on Today's Weather (YOU Need Historical Patterns).
#
#
#
# **understanding the Discount Factor (γ)**the Discount Factor Determines How Much You Care About Future Rewards:- **Γ = 0**: "I Only Care About Immediate Rewards" (very Myopic)- Example: Only Caring About This Month's Salary, Not Career Growth - **Γ = 0.9**: "future Rewards Are Worth 90% of Immediate Rewards"- Example: Investing Money - You Value Future Returns but Prefer Sooner - **Γ = 1**: "future Rewards Are as Valuable as Immediate Rewards"- Example: Climate Change Actions - Long-term Benefits Matter Equally**mathematical Impact**:- Return G*t = R*{T+1} + ΓR*{T+2} + Γ²R*{T+3} + ...- with Γ=0.9: G*t = R*{T+1} + 0.9×R*{T+2} + 0.81×R*{T+3} + ...- Future Rewards Get Progressively Less Important


```python
class GridWorld:
    """
    A simple GridWorld environment for demonstrating MDP concepts.
    
    The agent starts at (0,0) and tries to reach the goal at (3,3).
    There are obstacles and different reward structures.
    """
    
    def __init__(self, size=4, goal_reward=10, step_reward=-0.1, obstacle_reward=-5):
        self.size = size
        self.goal_reward = goal_reward
        self.step_reward = step_reward
        self.obstacle_reward = obstacle_reward
        
        # Define states, actions
        self.states = [(i, j) for i in range(size) for j in range(size)]
        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = {
            'up': (-1, 0),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1)
        }
        
        # Define special states
        self.start_state = (0, 0)
        self.goal_state = (3, 3)
        self.obstacles = [(1, 1), (2, 1), (1, 2)]  # Obstacle positions
        
        # Initialize transition probabilities and rewards
        self._build_transition_model()
        
    def _build_transition_model(self):
        """Build transition probability and reward models"""
        self.P = {}  # P[s][a] = [(prob, next_state, reward)]
        
        for state in self.states:
            self.P[state] = {}
            for action in self.actions:
                self.P[state][action] = self._get_transitions(state, action)
    
    def _get_transitions(self, state, action):
        """Get possible transitions for a state-action pair"""
        if state == self.goal_state:
            # Terminal state
            return [(1.0, state, 0)]
        
        if state in self.obstacles:
            # Can't take actions from obstacle states
            return [(1.0, state, self.obstacle_reward)]
        
        # Calculate intended next state
        dx, dy = self.action_effects[action]
        next_x, next_y = state[0] + dx, state[1] + dy
        
        # Check boundaries
        if (next_x < 0 or next_x >= self.size or 
            next_y < 0 or next_y >= self.size):
            next_state = state  # Stay in same state if hitting boundary
        else:
            next_state = (next_x, next_y)
        
        # Calculate reward
        if next_state == self.goal_state:
            reward = self.goal_reward
        elif next_state in self.obstacles:
            reward = self.obstacle_reward
        else:
            reward = self.step_reward
        
        # For simplicity, we'll use deterministic transitions
        # In practice, you might add noise (e.g., 0.8 prob intended direction,
        # 0.1 prob each perpendicular direction)
        return [(1.0, next_state, reward)]
    
    def get_valid_actions(self, state):
        """Get valid actions from a given state"""
        if state == self.goal_state or state in self.obstacles:
            return []
        return self.actions.copy()
    
    def visualize_grid(self, values=None, policy=None, title="GridWorld"):
        """Visualize the grid world with optional value function or policy"""
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Create grid
        grid = np.zeros((self.size, self.size))
        
        # Mark special states
        for i, j in self.obstacles:
            grid[i, j] = -1  # Obstacles
        
        goal_i, goal_j = self.goal_state
        grid[goal_i, goal_j] = 1  # Goal
        
        # Add values if provided
        if values is not None:
            for i in range(self.size):
                for j in range(self.size):
                    if (i, j) not in self.obstacles and (i, j) != self.goal_state:
                        grid[i, j] = values.get((i, j), 0)
        
        # Plot grid
        im = ax.imshow(grid, cmap='RdYlGn', aspect='equal')
        
        # Add text annotations
        for i in range(self.size):
            for j in range(self.size):
                if (i, j) == self.goal_state:
                    ax.text(j, i, 'G', ha='center', va='center', fontsize=16, fontweight='bold')
                elif (i, j) in self.obstacles:
                    ax.text(j, i, 'X', ha='center', va='center', fontsize=16, fontweight='bold')
                elif (i, j) == self.start_state:
                    ax.text(j, i, 'S', ha='center', va='center', fontsize=16, fontweight='bold')
                elif values is not None:
                    ax.text(j, i, f'{values.get((i, j), 0):.2f}', 
                           ha='center', va='center', fontsize=10)
        
        # Add policy arrows if provided
        if policy is not None:
            arrow_map = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
            for state, action in policy.items():
                if state not in self.obstacles and state != self.goal_state:
                    i, j = state
                    if action in arrow_map:
                        ax.text(j, i-0.3, arrow_map[action], ha='center', va='center', 
                               fontsize=12, fontweight='bold', color='blue')
        
        ax.set_xticks(range(self.size))
        ax.set_yticks(range(self.size))
        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.colorbar(im)
        plt.tight_layout()
        plt.show()

# Create and visualize the environment
env = GridWorld()
env.visualize_grid(title="GridWorld Environment\n(S=Start, G=Goal, X=Obstacles)")

print("GridWorld Environment Created!")
print(f"States: {len(env.states)}")
print(f"Actions: {env.actions}")
print(f"Start State: {env.start_state}")
print(f"Goal State: {env.goal_state}")
print(f"Obstacles: {env.obstacles}")
```


    
![png](CA2_files/CA2_6_0.png)
    


    GridWorld Environment Created!
    States: 16
    Actions: ['up', 'down', 'left', 'right']
    Start State: (0, 0)
    Goal State: (3, 3)
    Obstacles: [(1, 1), (2, 1), (1, 2)]


#
#
# 🎮 Understanding Our Gridworld Environmentbefore We Dive into the Code, Let's Understand What We're Building:
#
#
#
# **the Gridworld SETUP**```(0,0) → → → (0,3) ↓ X X ↓ ↓ X ◯ ↓ (3,0) → → → (3,3) 🎯```**legend:**- `S` at (0,0): Starting Position- `🎯` at (3,3): Goal (treasure!)- `X`: Obstacles (walls or Pits)- `◯`: Regular Empty Spaces- Arrows: Possible Movements
#
#
#
# **why This Environment Is Perfect for LEARNING**1. **small & Manageable**: 4×4 Grid = 16 States (easy to VISUALIZE)2. **clear Objective**: Get from Start to GOAL3. **interesting Obstacles**: Forces Strategic THINKING4. **deterministic**: Same Action Always Leads to Same Result (FOR Now)
#
#
#
# **reward Structure Explained**- **goal Reward (+10)**: Big Positive Reward for Reaching the Treasure- **step Penalty (-0.1)**: Small Negative Reward for Each Move (encourages Efficiency)- **obstacle Penalty (-5)**: Big Negative Reward for Hitting Obstacles (safety First!)**why These Specific Values?**- Goal Reward Is Much Larger Than Step Penalty → Encourages Reaching the Goal- Obstacle Penalty Is Significant → Discourages Dangerous Moves- Step Penalty Is Small → Prevents Infinite Wandering without Being Too Harsh
#
#
#
# **state Representation**each State Is a Tuple (row, Column):- (0,0) = Top-left Corner- (3,3) = Bottom-right Corner - States Are like Gps Coordinates for Our Agent

#
# Part 2: Policy Definition and Evaluation
#
#
# Exercise 2.1: Define Different Policies**definition:**a Policy Π(a|s) Defines the Probability of Taking Action a in State S. It's the Agent's Strategy for Choosing Actions.**mathematical Representation:**$$\pi(a|s) = P(\text{action} = a | \text{state} = S)$$**types of Policies:**- **deterministic Policy**: Π(a|s) ∈ {0, 1} - Always Chooses the Same Action in a Given State- **stochastic Policy**: Π(a|s) ∈ [0, 1] - Chooses Actions Probabilistically**policies We'll IMPLEMENT:**1. **random Policy**: Equal Probability for All Valid ACTIONS2. **greedy Policy**: Always Move towards the Goal 3. **custom Policy**: Your Own Strategic Policy---
#
#
# Exercise 2.2: Policy Evaluation**definition:**policy Evaluation Computes the Value Function V^π(s) for a Given Policy Π. It Answers: "HOW Good Is This Policy?"**iterative Policy Evaluation ALGORITHM:**1. **initialize**: V(s) = 0 for All States S2. **repeat until Convergence**:- for Each State S:- V*new(s) = Σ*a Π(a|s) Σ*{s'} P(s'|s,a)[r(s,a,s') + ΓV(S')]3. **return**: Converged Value Function V**convergence Condition:**max*s |v*new(s) - V*old(s)| < Θ (where Θ Is a Small Threshold, E.g., 1E-6)**INTUITION:**WE Start with All State Values at Zero and Iteratively Update Them Based on the Bellman Equation until They Stabilize. It's like Repeatedly Asking "IF I Follow This Policy, How Much Reward Will I Get?" until the Answer Stops Changing.

#
#
# 🧭 Policy Deep Dive: Understanding Different Strategies**what Is a Policy?**a Policy Is like a Gps Navigation System for Our Agent. It Tells the Agent What to Do in Every Possible Situation.**mathematical Definition:**π(a|s) = Probability of Taking Action a When in State S---
#
#
# 📋 Types of Policies We'll IMPLEMENT**1. Random Policy** 🎲**strategy:** "when in Doubt, Flip a Coin"**mathematical Definition:** Π(a|s) = 1/|VALID_ACTIONS| for All Valid Actions**example:** at State (1,0), If We Can Go [UP, Down, Right], Each Has 33.33% Probability**advantages:**- Explores All Possibilities Equally- Simple to Implement- Guarantees Exploration**disadvantages:**- Not Very Efficient- like Wandering Randomly in a Maze- No Learning from EXPERIENCE---**2. Greedy Policy** 🎯**strategy:** "always Move Closer to the Goal"**mathematical Definition:** Π(a|s) = 1 If a Minimizes Distance to Goal, 0 Otherwise**example:** at State (1,0), If Goal Is at (3,3), Prefer "down" and "right"**advantages:**- Very Efficient When It Works- Direct Path to Goal- Fast Convergence**disadvantages:**- Can Get Stuck in Local Optima- Might Walk into Obstacles- No Exploration of Alternative PATHS---**3. Custom Policy** 🎨**strategy:** Your Creative Combination of Strategies**examples:**- **epsilon-greedy**: 90% Greedy, 10% Random- **safety-first**: Avoid Actions That Lead near Obstacles- **wall-follower**: Stay Close to Boundaries---
#
#
# 🎮 Real-world Analogies**policy Vs Strategy in Games:**think of Different Video Game Playing Styles:- **aggressive Player**: Always Attacks (deterministic Policy)- **defensive Player**: Always Defends (deterministic Policy)- **adaptive Player**: 70% Attack, 30% Defend (stochastic Policy)**why Stochastic Policies?**sometimes Randomness Helps:- **exploration**: Discover New Paths You Wouldn't Normally Try- **unpredictability**: in Competitive Games, Being Predictable Is Bad- **robustness**: Handle Uncertainty in the Environment**restaurant Choice Analogy:**- **random Policy**: Pick Restaurants Randomly- **greedy Policy**: Always Go to Your Current Favorite- **epsilon-greedy Policy**: Usually Go to Favorite, Sometimes Try Something New


```python
class Policy:
    """Base class for policies"""
    
    def __init__(self, env):
        self.env = env
        
    def get_action_prob(self, state, action):
        """Return probability of taking action in state"""
        raise NotImplementedError
    
    def get_action_probs(self, state):
        """Return dictionary of action probabilities for state"""
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions:
            return {}
        
        probs = {}
        for action in valid_actions:
            probs[action] = self.get_action_prob(state, action)
        return probs
    
    def select_action(self, state):
        """Select an action according to policy"""
        probs = self.get_action_probs(state)
        if not probs:
            return None
        
        actions = list(probs.keys())
        probabilities = list(probs.values())
        return np.random.choice(actions, p=probabilities)


class RandomPolicy(Policy):
    """Random policy - equal probability for all valid actions"""
    
    def get_action_prob(self, state, action):
        valid_actions = self.env.get_valid_actions(state)
        if action in valid_actions:
            return 1.0 / len(valid_actions)
        return 0.0


class GreedyPolicy(Policy):
    """Greedy policy - always move towards goal"""
    
    def get_action_prob(self, state, action):
        if state == self.env.goal_state or state in self.env.obstacles:
            return 0.0
        
        # Calculate Manhattan distance to goal for each action
        goal_x, goal_y = self.env.goal_state
        current_x, current_y = state
        
        best_actions = []
        min_distance = float('inf')
        
        for act in self.env.get_valid_actions(state):
            dx, dy = self.env.action_effects[act]
            next_x, next_y = current_x + dx, current_y + dy
            
            # Check if next state is valid
            if (0 <= next_x < self.env.size and 0 <= next_y < self.env.size 
                and (next_x, next_y) not in self.env.obstacles):
                
                distance = abs(next_x - goal_x) + abs(next_y - goal_y)
                
                if distance < min_distance:
                    min_distance = distance
                    best_actions = [act]
                elif distance == min_distance:
                    best_actions.append(act)
        
        if action in best_actions:
            return 1.0 / len(best_actions)
        return 0.0


# Test the policies
random_policy = RandomPolicy(env)
greedy_policy = GreedyPolicy(env)

# Test random policy
print("Random Policy Example:")
test_state = (1, 0)
print(f"State: {test_state}")
print(f"Valid actions: {env.get_valid_actions(test_state)}")
print(f"Action probabilities: {random_policy.get_action_probs(test_state)}")

print("\nGreedy Policy Example:")
print(f"State: {test_state}")
print(f"Action probabilities: {greedy_policy.get_action_probs(test_state)}")

# Test action selection
print(f"\nSelected actions from state {test_state}:")
print(f"Random policy: {[random_policy.select_action(test_state) for _ in range(5)]}")
print(f"Greedy policy: {[greedy_policy.select_action(test_state) for _ in range(5)]}")
```

    Random Policy Example:
    State: (1, 0)
    Valid actions: ['up', 'down', 'left', 'right']
    Action probabilities: {'up': 0.25, 'down': 0.25, 'left': 0.25, 'right': 0.25}
    
    Greedy Policy Example:
    State: (1, 0)
    Action probabilities: {'up': 0.0, 'down': 1.0, 'left': 0.0, 'right': 0.0}
    
    Selected actions from state (1, 0):
    Random policy: [np.str_('down'), np.str_('right'), np.str_('left'), np.str_('left'), np.str_('up')]
    Greedy policy: [np.str_('down'), np.str_('down'), np.str_('down'), np.str_('down'), np.str_('down')]



```python
def policy_evaluation(env, policy, gamma=0.9, theta=1e-6, max_iterations=1000):
    """
    Iterative policy evaluation to compute state-value function V^π(s)
    
    Args:
        env: GridWorld environment
        policy: Policy object
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
    
    Returns:
        V: Dictionary mapping states to values
    """
    # Initialize value function
    V = {state: 0.0 for state in env.states}
    
    for iteration in range(max_iterations):
        delta = 0.0
        V_new = V.copy()
        
        for state in env.states:
            if state == env.goal_state:
                continue  # Terminal state, value remains 0
                
            # Calculate new value
            v_old = V[state]
            v_new = 0.0
            
            # Get action probabilities for this state
            action_probs = policy.get_action_probs(state)
            
            for action, prob in action_probs.items():
                # Get transitions for this state-action pair
                transitions = env.P[state][action]
                
                for trans_prob, next_state, reward in transitions:
                    v_new += prob * trans_prob * (reward + gamma * V[next_state])
            
            V_new[state] = v_new
            delta = max(delta, abs(v_old - v_new))
        
        V = V_new
        
        # Check for convergence
        if delta < theta:
            print(f"Policy evaluation converged after {iteration + 1} iterations")
            break
    else:
        print(f"Policy evaluation stopped after {max_iterations} iterations")
    
    return V


def evaluate_and_visualize_policy(env, policy, policy_name, gamma=0.9):
    """Evaluate a policy and visualize the results"""
    print(f"\n{'='*50}")
    print(f"Evaluating {policy_name}")
    print('='*50)
    
    # Evaluate policy
    V = policy_evaluation(env, policy, gamma=gamma)
    
    # Print some sample values
    print(f"\nSample State Values:")
    sample_states = [(0,0), (1,0), (2,0), (3,0), (0,3), (2,2)]
    for state in sample_states:
        if state in V:
            print(f"V({state}) = {V[state]:.3f}")
    
    # Create policy dictionary for visualization
    policy_dict = {}
    for state in env.states:
        if state != env.goal_state and state not in env.obstacles:
            action_probs = policy.get_action_probs(state)
            if action_probs:
                # Get most likely action
                best_action = max(action_probs.items(), key=lambda x: x[1])[0]
                policy_dict[state] = best_action
    
    # Visualize
    env.visualize_grid(values=V, policy=policy_dict, 
                      title=f"{policy_name}\nState Values and Policy")
    
    return V


# Evaluate different policies
print("Starting Policy Evaluation...")

# Random Policy
V_random = evaluate_and_visualize_policy(env, random_policy, "Random Policy")

# Greedy Policy  
V_greedy = evaluate_and_visualize_policy(env, greedy_policy, "Greedy Policy")
```

    Starting Policy Evaluation...
    
    ==================================================
    Evaluating Random Policy
    ==================================================
    Policy evaluation converged after 51 iterations
    
    Sample State Values:
    V((0, 0)) = -3.141
    V((1, 0)) = -3.617
    V((2, 0)) = -3.427
    V((3, 0)) = -2.299
    V((0, 3)) = -2.299
    V((2, 2)) = -1.576



    
![png](CA2_files/CA2_11_1.png)
    


    
    ==================================================
    Evaluating Greedy Policy
    ==================================================
    Policy evaluation converged after 7 iterations
    
    Sample State Values:
    V((0, 0)) = 5.495
    V((1, 0)) = 6.217
    V((2, 0)) = 7.019
    V((3, 0)) = 7.910
    V((0, 3)) = 7.910
    V((2, 2)) = 8.900



    
![png](CA2_files/CA2_11_3.png)
    


#
#
# 🔍 Understanding Policy Evaluation Step-by-steppolicy Evaluation Answers the Question: **"how Good Is Each State If I Follow This Policy?"**
#
#
#
# **the Intuition**imagine You're Evaluating Different Starting Positions in a Board Game:- Some Positions Are Naturally Better (closer to Winning)- Some Positions Are Worse (closer to Losing) - the "value" of a Position Depends on How Well You'll Do from There
#
#
#
# **mathematical Breakdown****the Bellman Equation for State Values:**```v^π(s) = Σ*a Π(a|s) × Σ*{s'} P(s'|s,a) × [r(s,a,s') + Γ × V^π(s')]```**let's Decode This Step by STEP:**1. **for Each Possible Action A**: Π(a|s) = "HOW Likely Am I to Take Action a in State S?"2. **for Each Possible Next State S'**: P(s'|s,a) = "IF I Take Action A, What's the Chance I End Up in State S'?"3. **calculate Immediate Reward + Future Value**: R(s,a,s') + Γ × V^π(s')- R(s,a,s') = "what Reward Do I Get Immediately?"- Γ × V^π(s') = "what's the Discounted Future VALUE?"4. **sum Everything Up**: This Gives the Expected Value of Being in State S
#
#
#
# **simple Example**let's Say We're at State (2,2) with a Random Policy:```python
# Random Policy: Equal Probability for All Valid Actionsπ(up|s) = 0.25, Π(down|s) = 0.25, Π(left|s) = 0.25, Π(right|s) = 0.25
# FOR Action "UP" → Next State (1,2)CONTRIBUTION*UP = 0.25 × 1.0 × (-0.1 + 0.9 × V(1,2))
# FOR Action "down" → Next State (3,2)CONTRIBUTION*DOWN = 0.25 × 1.0 × (-0.1 + 0.9 × V(3,2))
# ... and So on for Left and RIGHTV(2,2) = Contribution*up + Contribution*down + Contribution*left + Contribution*right```
#
#
#
# **why Iterative?**- We Start with V(s) = 0 for All States (initial Guess)- Each Iteration Improves Our Estimate Using Current Values- Eventually, Values Converge to True Values- like Asking "IF I Knew the Value of My Neighbors, What Would My Value Be?"
#
#
#
# **convergence Intuition**think of It like Gossip Spreading in a Neighborhood:- Initially, Nobody Knows the True "gossip" (values)- Each Iteration, Neighbors Share Information - Eventually, Everyone Converges to the Same True Story

#
#
# Exercise 2.3: Create Your Custom Policy**task**: Design and Implement Your Own Policy. Consider Strategies Like:- **wall-following**: Try to Stay Close to Walls- **risk-averse**: Avoid Obstacles with Higher Probability- **exploration-focused**: Balance between Moving towards Goal and Exploring**your Implementation Below**:


```python
class CustomPolicy(Policy):
    """
    TODO: Implement your custom policy here
    
    Ideas:
    - Combine greedy behavior with some exploration
    - Avoid states near obstacles
    - Use different strategies for different regions of the grid
    """
    
    def __init__(self, env, exploration_prob=0.2):
        super().__init__(env)
        self.exploration_prob = exploration_prob
    
    def get_action_prob(self, state, action):
        """
        Implement custom policy: epsilon-greedy with obstacle avoidance
        
        Strategy:
        1. With probability (1-exploration_prob), prefer greedy actions
        2. With probability exploration_prob, choose randomly among valid actions
        3. Apply penalty for actions that lead near obstacles
        """
        if state == self.env.goal_state or state in self.env.obstacles:
            return 0.0
        
        valid_actions = self.env.get_valid_actions(state)
        if action not in valid_actions:
            return 0.0
        
        # Calculate base probabilities for each action
        goal_x, goal_y = self.env.goal_state
        current_x, current_y = state
        
        # Calculate next position
        dx, dy = self.env.action_effects[action]
        next_x, next_y = current_x + dx, current_y + dy
        
        # Check if next state is valid
        if not (0 <= next_x < self.env.size and 0 <= next_y < self.env.size):
            return 0.0
            
        # Calculate distance to goal from next state
        goal_distance = abs(next_x - goal_x) + abs(next_y - goal_y)
        current_distance = abs(current_x - goal_x) + abs(current_y - goal_y)
        
        # Check if action moves closer to goal
        moves_towards_goal = goal_distance < current_distance
        
        # Check if next state is near obstacles
        near_obstacle = False
        if (next_x, next_y) in self.env.obstacles:
            return 0.0  # Can't move to obstacle
        
        for obs_x, obs_y in self.env.obstacles:
            if abs(next_x - obs_x) + abs(next_y - obs_y) <= 1:
                near_obstacle = True
                break
        
        # Calculate probability based on greedy behavior and exploration
        if moves_towards_goal and not near_obstacle:
            # Best case: moves towards goal and safe
            return (1 - self.exploration_prob) * 0.7 + self.exploration_prob / len(valid_actions)
        elif moves_towards_goal and near_obstacle:
            # Good direction but risky
            return (1 - self.exploration_prob) * 0.2 + self.exploration_prob / len(valid_actions)
        elif not moves_towards_goal and not near_obstacle:
            # Safe but not optimal direction
            return (1 - self.exploration_prob) * 0.1 + self.exploration_prob / len(valid_actions)
        else:
            # Worst case: away from goal and risky
            return self.exploration_prob / len(valid_actions)
    
    def get_action_probs(self, state):
        """Return normalized action probabilities"""
        probs = super().get_action_probs(state)
        
        # Normalize probabilities to ensure they sum to 1
        if probs:
            total = sum(probs.values())
            if total > 0:
                probs = {action: prob / total for action, prob in probs.items()}
        
        return probs

# TODO: Test your custom policy
custom_policy = CustomPolicy(env, exploration_prob=0.3)

# Test the custom policy
test_state = (2, 0)
print(f"Custom Policy at state {test_state}:")
print(f"Action probabilities: {custom_policy.get_action_probs(test_state)}")

# Evaluate the custom policy
V_custom = evaluate_and_visualize_policy(env, custom_policy, "Custom Policy")
```

    Custom Policy at state (2, 0):
    Action probabilities: {'up': 0.11718750000000001, 'down': 0.8828125, 'left': 0.0, 'right': 0.0}
    
    ==================================================
    Evaluating Custom Policy
    ==================================================
    Policy evaluation converged after 46 iterations
    
    Sample State Values:
    V((0, 0)) = 3.587
    V((1, 0)) = 4.096
    V((2, 0)) = 5.388
    V((3, 0)) = 6.364
    V((0, 3)) = 6.364
    V((2, 2)) = 8.367



    
![png](CA2_files/CA2_14_1.png)
    


#
# Part 3: Action-value Functions (q-functions)
#
#
# Exercise 3.1: Computing Q-values**definition:**the Action-value Function Q^π(s,a) Represents the Expected Return When Taking Action a in State S and Then Following Policy Π.**key Question Q-functions Answer:**q-functions Answer: "what If I Take This Specific Action Here, Then Follow My Policy?"**mathematical Relationships:****v from Q (policy-weighted Average):**$$v^π(s) = \sum*a Π(a|s) Q^π(s,a)$$**q from V (bellman Backup):**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**bellman Equation for Q:**$$q^π(s,a) = \sum*{s'} P(s'|s,a)[r(s,a,s') + Γ \sum*{a'} Π(a'|s')q^π(s',a')]$$**intuition:**- **v(s)**: "HOW Good Is This State?" (following Current Policy)- **q(s,a)**: "HOW Good Is This Specific Action?" (then Following Policy)the V-q Relationship Is like Asking:- V: "HOW Well Will I Do from This Chess Position?"- Q: "HOW Well Will I Do If I Move My Queen Here, Then Play Normally?"

#
#
# 🎯 Q-functions Deep Dive: the "what If" Values**core Concept:**q-functions Provide Action-specific Evaluations, Allowing Us to Compare Different Choices Directly.---
#
#
# 🍕 Restaurant Decision Analogy**scenario:** You're Choosing a Restaurant from Downtown Location.**value Functions:**- **v(downtown)** = 7.5 → "average Satisfaction from This Location with My Usual Choices"- **q(downtown, Pizza*place)** = 8.2 → "satisfaction If I Specifically Choose Pizza"- **q(downtown, Sushi*place)** = 6.8 → "satisfaction If I Specifically Choose Sushi"- **q(downtown, Burger*place)** = 7.1 → "satisfaction If I Specifically Choose Burgers"**policy Calculation:**if My Policy Is 50% Pizza, 30% Sushi, 20% Burgers:v(downtown) = 0.5×8.2 + 0.3×6.8 + 0.2×7.1 = 4.1 + 2.04 + 1.42 = 7.56 ✓---
#
#
# 🧮 Mathematical Relationships EXPLAINED**1. V from Q (weighted Average):**$$v^π(s) = \sum*a Π(a|s) × Q^π(s,a)$$**interpretation:** State Value = Probability of Each Action × Value of That ACTION**2. Q from V (bellman Backup):**$$q^π(s,a) = \sum*{s'} P(s'|s,a) × [r(s,a,s') + Γv^π(s')]$$**interpretation:** Action Value = Immediate Reward + Discounted Future State Value---
#
#
# 🔥 Why Q-functions MATTER**1. Direct Action Comparison:**- Q(s, Left) = 5.2 Vs Q(s, Right) = 7.8 → Choose Right!- No Need to Compute State Values FIRST**2. Policy Improvement:**- Π*new(s) = Argmax*a Q^π*old(s,a)- Directly Find the Best ACTION**3. Optimal Decision Making:**- Q*(s,a) Tells Us the Value of Each Action under Optimal Behavior- Essential for Q-learning Algorithms---
#
#
# 📊 Visual Understandingthink of Q-values as Action-specific "heat Maps":- **hot Spots** (high Q-values): Good Actions to Take- **cold Spots** (LOW Q-values): Actions to Avoid- **separate Map for Each Action**: Q(s,↑), Q(s,↓), Q(s,←), Q(s,→)**gridworld Example:**- Q(state, "toward*goal") Typically Has Higher Values- Q(state, "toward*obstacle") Typically Has Lower Values- Q(state, "toward*wall") Often Has Negative Values - Like: "restaurant Satisfaction = Meal Quality + How I'll Feel Tomorrow"
#
#
#
# **why Q-functions MATTER**1. **better Decision Making**: Q-values Directly Tell Us Which Action Is Best- Max*a Q(s,a) Gives the Best Action in State S2. **policy Improvement**: We Can Improve Policies by Being Greedy W.r.t. Q-values- Π*new(s) = Argmax*a Q^Π_OLD(S,A)3. **action Comparison**: Compare Different Actions in the Same State- "should I Go Left or Right from Here?"
#
#
#
# **visual Understanding**think of Q-values as a "heat Map" for Each Action:- **hot Spots** (high Q-values): Good Actions to Take- **cold Spots** (LOW Q-values): Actions to Avoid - **different Maps for Each Action**: Q(s,up), Q(s,down), Q(s,left), Q(s,right)
#
#
#
# **common Confusion: V Vs Q**- **v(s)**: "HOW Good Is My Current Strategy from This Position?"- **q(s,a)**: "HOW Good Is This Specific Move, Then Using My Strategy?"it's like Asking:- V: "HOW Well Will I Do in This Chess Position?" - Q: "HOW Well Will I Do If I Move My Queen Here, Then Play Normally?"


```python
def compute_q_from_v(env, V, gamma=0.9):
    """
    Compute Q-values from state values using:
    Q(s,a) = Σ_{s'} P(s'|s,a)[R(s,a,s') + γV(s')]
    """
    Q = {}
    
    for state in env.states:
        Q[state] = {}
        for action in env.actions:
            if state == env.goal_state:
                Q[state][action] = 0.0
                continue
                
            q_value = 0.0
            transitions = env.P[state][action]
            
            for prob, next_state, reward in transitions:
                q_value += prob * (reward + gamma * V[next_state])
            
            Q[state][action] = q_value
    
    return Q


def compute_v_from_q(env, Q, policy):
    """
    Compute state values from Q-values using:
    V(s) = Σ_a π(a|s) Q(s,a)
    """
    V = {}
    
    for state in env.states:
        if state == env.goal_state:
            V[state] = 0.0
            continue
            
        v_value = 0.0
        action_probs = policy.get_action_probs(state)
        
        for action, prob in action_probs.items():
            v_value += prob * Q[state][action]
        
        V[state] = v_value
    
    return V


def visualize_q_values(env, Q, title="Q-Values"):
    """Visualize Q-values for all state-action pairs"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    actions = ['up', 'down', 'left', 'right']
    
    for idx, action in enumerate(actions):
        ax = axes[idx // 2, idx % 2]
        
        # Create grid for this action
        q_grid = np.zeros((env.size, env.size))
        
        for i in range(env.size):
            for j in range(env.size):
                state = (i, j)
                if state in Q and action in Q[state]:
                    q_grid[i, j] = Q[state][action]
        
        # Plot
        im = ax.imshow(q_grid, cmap='RdYlGn', aspect='equal')
        ax.set_title(f'Q-values for "{action}" action')
        
        # Add text annotations
        for i in range(env.size):
            for j in range(env.size):
                state = (i, j)
                if state == env.goal_state:
                    ax.text(j, i, 'G', ha='center', va='center', 
                           fontsize=16, fontweight='bold')
                elif state in env.obstacles:
                    ax.text(j, i, 'X', ha='center', va='center', 
                           fontsize=16, fontweight='bold')
                else:
                    if state in Q and action in Q[state]:
                        ax.text(j, i, f'{Q[state][action]:.2f}', 
                               ha='center', va='center', fontsize=8)
        
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# Compute Q-values for the greedy policy
print("Computing Q-values for Greedy Policy...")
Q_greedy = compute_q_from_v(env, V_greedy, gamma=0.9)

# Verify V-Q relationship
V_from_Q = compute_v_from_q(env, Q_greedy, greedy_policy)

print("Verification of V-Q relationship:")
print("State\t\tV(direct)\tV(from Q)\tDifference")
print("-" * 60)
for state in [(0,0), (1,0), (2,0), (0,1)]:
    if state in V_greedy:
        diff = abs(V_greedy[state] - V_from_Q[state])
        print(f"{state}\t\t{V_greedy[state]:.4f}\t\t{V_from_Q[state]:.4f}\t\t{diff:.6f}")

# Visualize Q-values
visualize_q_values(env, Q_greedy, "Q-Values for Greedy Policy")
```

    Computing Q-values for Greedy Policy...
    Verification of V-Q relationship:
    State		V(direct)	V(from Q)	Difference
    ------------------------------------------------------------
    (0, 0)		5.4954		5.4954		0.000000
    (1, 0)		6.2171		6.2171		0.000000
    (2, 0)		7.0190		7.0190		0.000000
    (0, 1)		6.2171		6.2171		0.000000



    
![png](CA2_files/CA2_17_1.png)
    


#
# Part 4: Policy Improvement and Policy Iteration
#
#
# Exercise 4.1: Policy Improvement**definition:**given a Value Function V^π, We Can Improve the Policy by Being Greedy with Respect to the Action-value Function.**policy Improvement Formula:**$$π'(s) = \arg\max*a Q^π(s,a) = \arg\max*a \sum*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]$$**interpretation:** Choose the Action That Maximizes Expected Return from Each State.**policy Improvement Theorem:**if Π' Is Greedy with Respect to V^π, Then V^π'(s) ≥ V^π(s) for All States S.**translation:** "IF I Always Choose the Best Action Based on My Current Understanding, I Can Only Do Better (OR at Least as Well)."---
#
#
# Exercise 4.2: Policy Iteration Algorithm**policy Iteration STEPS:**1. **initialize**: Start with Arbitrary Policy Π₀2. **repeat until Convergence**:- **policy Evaluation**: Compute V^π*k (solve Bellman Equation)- **policy Improvement**: Π*{K+1}(S) = Argmax*a Q^Π_K(S,A)3. **output**: Optimal Policy Π* and Value Function V***convergence Guarantee:** Policy Iteration Is Guaranteed to Converge to the Optimal Policy in Finite Time for Finite Mdps.**why It Works:**- Each Step Produces a Better (OR Equal) Policy- There Are Only Finitely Many Deterministic Policies- Must Eventually Reach Optimal Policy

#
#
# 🚀 Policy Improvement Deep Dive: Making Better Decisions**core Idea:** Use the Value Function to Make Better Action Choices.---
#
#
# 📚 Learning Process Analogy**scenario:** You're Learning to Play Chess.**policy Evaluation:** "HOW Good Is My Current Playing Style?"- Analyze Your Current Strategy- Evaluate Typical Game Outcomes- Identify Strengths and Weaknesses**policy Improvement:** "HOW Can I Play Better?"- Look at Each Position Where You Made Suboptimal Moves- Replace Bad Moves with Better Alternatives- Update Your Playing Strategy**policy Iteration:** Repeat This Cycle until You Can't Improve Further.---
#
#
# 🧮 Mathematical Foundations**policy Improvement Theorem:**if Π' Is Greedy W.r.t. V^π, Then V^π'(s) ≥ V^π(s) for All S.**proof INTUITION:**1. **greedy Action**: Choose a Such That Q^π(s,a) Is MAXIMIZED2. **definition**: Q^π(s,a) ≥ V^π(s) for the Chosen ACTION3. **new Policy**: Π'(s) Gives This Optimal ACTION4. **result**: V^π'(s) ≥ V^π(s)**why Greedy Improvement Works:**- Current Policy Chooses Actions with Average Value V^π(s)- Greedy Policy Chooses Action with Maximum Value Q^π(s,a)- Maximum ≥ Average, So New Policy Is Better---
#
#
# 🔄 Policy Iteration: the Complete Cycle**step 1 - Policy Evaluation:** "HOW Good Is My Current Policy?"```v^π(s) ← Expected Return Following Π from State S```**step 2 - Policy Improvement:** "what's the Best Action in Each State?"```π'(s) ← Action That Maximizes Q^π(s,a)```**step 3 - Check Convergence:** "DID My Policy Change?"```if Π'(s) = Π(s) for All S: Stop (optimal Found)else: Π ← Π' and Repeat```---
#
#
# 🎯 Key PROPERTIES**1. Monotonic IMPROVEMENT:**V^Π₀ ≤ V^Π₁ ≤ V^Π₂ ≤ ... ≤ V^Π**2. Finite Convergence:**algorithm Terminates in Finite Steps (FOR Finite MDPS)**3. Optimal Solution:**final Policy Π* Is Optimal: V^π* = V**4. Model-based:**requires Knowledge of Transition Probabilities P(s'|s,a) and Rewards R(s,a,s')think of a Student Improving Their Study STRATEGY:1. **current Strategy** (policy Π): "I Study Randomly for 2 HOURS"2. **evaluate Strategy** (policy Evaluation): "HOW Well Does This Work for Each Subject?" 3. **find Better Strategy** (policy Improvement): "math Needs 3 Hours, History Needs 1 HOUR"4. **repeat**: Keep Refining until No More Improvements Possible
#
#
#
# **mathematical Intuition**policy Improvement Theorem**: If Q^π(s,a) > V^π(s) for Some Action A, Then Taking Action a Is Better Than Following Policy Π.**translation**: "IF Doing Action a Gives Higher Value Than My Current Average, I Should Do Action a More Often!"**greedy Improvement**:```pythonπ*new(s) = Argmax*a Q^π(s,a)```"always Choose the Action with Highest Q-value"
#
#
#
# **why Does This Work?**monotonic Improvement**: Each Policy Improvement Step Makes the Policy at Least as Good, Usually Better.**proof Sketch**:- If We're Greedy W.r.t. Q^π, We Get V^π_new ≥ V^π- "IF I Always Choose the Best Available Action, I Can't Do Worse"
#
#
#
# **policy Iteration: the Complete Algorithm**the Cycle**:```random Policy → Evaluate → Improve → Evaluate → Improve → ... → Optimal Policy```**why It CONVERGES**:1. **finite State/action Space**: Limited Number of Possible POLICIES2. **monotonic Improvement**: Each Step Makes Policy Better (OR SAME)3. **NO Cycles**: Can't Go Backwards to a Worse POLICY4. **must Terminate**: Eventually Reach Optimal Policy
#
#
#
# **real-world Example: Learning to Drive**iteration 1**:- **policy**: "drive Slowly Everywhere" - **evaluation**: "safe but Inefficient on Highways"- **improvement**: "drive Fast on Highways, Slow in Neighborhoods"**iteration 2**:- **policy**: "speed Varies by Road Type"- **evaluation**: "good, but Inefficient in Traffic" - **improvement**: "also Consider Traffic Conditions"**final Policy**: "optimal Speed Based on Road Type, Traffic, Weather, Etc."
#
#
#
# **key INSIGHTS**1. **guaranteed Improvement**: Policy Iteration Always Finds the Optimal Policy (FOR Finite MDPS)2. **fast Convergence**: Usually Converges in Just a Few ITERATIONS3. **NO Exploration Needed**: Uses Complete Model Knowledge (unlike Q-learning LATER)4. **computational Cost**: Each Iteration Requires Solving the Bellman Equation
#
#
#
# **common Pitfalls**- **getting Stuck**: in Stochastic Environments, Might Need Exploration- **computational Cost**: Policy Evaluation Can Be Expensive - **model Required**: Need to Know P(s'|s,a) and R(s,a,s')


```python
class GreedyActionPolicy(Policy):
    """Policy that is greedy with respect to given Q-values"""
    
    def __init__(self, env, Q):
        super().__init__(env)
        self.Q = Q
    
    def get_action_prob(self, state, action):
        if state == self.env.goal_state or state in self.env.obstacles:
            return 0.0
        
        valid_actions = self.env.get_valid_actions(state)
        if not valid_actions or action not in valid_actions:
            return 0.0
        
        # Find the best action(s)
        best_value = max(self.Q[state][a] for a in valid_actions)
        best_actions = [a for a in valid_actions if abs(self.Q[state][a] - best_value) < 1e-10]
        
        if action in best_actions:
            return 1.0 / len(best_actions)
        return 0.0


def policy_improvement(env, V, gamma=0.9):
    """
    Improve policy by being greedy with respect to value function
    Returns the new policy and whether it's different from greedy policy based on V
    """
    # Compute Q-values from V
    Q = compute_q_from_v(env, V, gamma)
    
    # Create greedy policy
    improved_policy = GreedyActionPolicy(env, Q)
    
    return improved_policy, Q


def policy_iteration(env, initial_policy=None, gamma=0.9, max_iterations=100):
    """
    Policy iteration algorithm
    
    Returns:
        - Final policy
        - List of value functions for each iteration
        - List of policies for each iteration
    """
    if initial_policy is None:
        initial_policy = RandomPolicy(env)
    
    policy = initial_policy
    V_history = []
    policy_history = []
    
    print("Starting Policy Iteration...")
    print("=" * 50)
    
    for iteration in range(max_iterations):
        print(f"\nIteration {iteration + 1}:")
        
        # Policy Evaluation
        print("  - Policy Evaluation...")
        V = policy_evaluation(env, policy, gamma=gamma, theta=1e-8)
        V_history.append(V.copy())
        
        # Policy Improvement
        print("  - Policy Improvement...")
        new_policy, Q = policy_improvement(env, V, gamma)
        
        # Check if policy has converged by comparing action selections
        policy_changed = False
        for state in env.states:
            if state != env.goal_state and state not in env.obstacles:
                old_probs = policy.get_action_probs(state)
                new_probs = new_policy.get_action_probs(state)
                
                # Check if the greedy actions are different
                if old_probs and new_probs:
                    old_best = max(old_probs.items(), key=lambda x: x[1])[0]
                    new_best = max(new_probs.items(), key=lambda x: x[1])[0]
                    if old_best != new_best:
                        policy_changed = True
                        break
        
        policy_history.append(policy)
        
        if not policy_changed:
            print(f"\n  ✓ Policy converged after {iteration + 1} iterations!")
            break
        else:
            print("  → Policy changed, continuing...")
            policy = new_policy
    
    else:
        print(f"\n  ⚠ Maximum iterations ({max_iterations}) reached")
    
    return new_policy, V_history, policy_history


# Run Policy Iteration starting from Random Policy
print("Policy Iteration Experiment:")
print("Starting from Random Policy")

optimal_policy, V_history, policy_history = policy_iteration(
    env, RandomPolicy(env), gamma=0.9, max_iterations=10
)

print(f"\nPolicy Iteration completed!")
print(f"Total iterations: {len(V_history)}")

# Visualize the final optimal policy
final_V = V_history[-1]

# Create policy dictionary for visualization
optimal_policy_dict = {}
for state in env.states:
    if state != env.goal_state and state not in env.obstacles:
        action_probs = optimal_policy.get_action_probs(state)
        if action_probs:
            best_action = max(action_probs.items(), key=lambda x: x[1])[0]
            optimal_policy_dict[state] = best_action

env.visualize_grid(values=final_V, policy=optimal_policy_dict, 
                  title="Optimal Policy from Policy Iteration\nState Values and Optimal Actions")
```

    Policy Iteration Experiment:
    Starting from Random Policy
    Starting Policy Iteration...
    ==================================================
    
    Iteration 1:
      - Policy Evaluation...
    Policy evaluation converged after 67 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 2:
      - Policy Evaluation...
    Policy evaluation converged after 154 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 3:
      - Policy Evaluation...
    Policy evaluation converged after 31 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 4:
      - Policy Evaluation...
    Policy evaluation converged after 7 iterations
      - Policy Improvement...
    
      ✓ Policy converged after 4 iterations!
    
    Policy Iteration completed!
    Total iterations: 4



    
![png](CA2_files/CA2_20_1.png)
    



```python
# Let's add a simple demonstration to understand policy iteration better

def simple_policy_iteration_demo(env, gamma=0.9):
    """
    A simplified demonstration of how policy iteration works
    This version shows the key steps more clearly
    """
    print("🎯 POLICY ITERATION DEMONSTRATION")
    print("=" * 50)
    
    # Step 1: Start with a random policy
    current_policy = RandomPolicy(env)
    iteration = 0
    
    print(f"\n📍 Starting with Random Policy")
    
    while iteration < 5:  # Limit to 5 iterations for demo
        iteration += 1
        print(f"\n🔄 ITERATION {iteration}")
        print("-" * 30)
        
        # Step 2: Policy Evaluation - "How good is my current policy?"
        print("📊 Step 1: Evaluating current policy...")
        current_values = policy_evaluation(env, current_policy, gamma=gamma, theta=1e-6)
        
        # Show a few sample values
        sample_states = [(0,0), (1,0), (2,0)]
        print("   Sample state values:")
        for state in sample_states:
            if state in current_values:
                print(f"   V({state}) = {current_values[state]:.3f}")
        
        # Step 3: Policy Improvement - "Can I do better?"
        print("🚀 Step 2: Improving policy based on values...")
        
        # Compute Q-values
        Q_values = compute_q_from_v(env, current_values, gamma)
        
        # Create improved policy (greedy w.r.t. Q-values)
        improved_policy = GreedyActionPolicy(env, Q_values)
        
        # Step 4: Check if policy changed
        policy_changed = False
        changes_count = 0
        
        for state in env.states:
            if state != env.goal_state and state not in env.obstacles:
                old_action_probs = current_policy.get_action_probs(state)
                new_action_probs = improved_policy.get_action_probs(state)
                
                if old_action_probs and new_action_probs:
                    # Get the most likely actions
                    old_best = max(old_action_probs.items(), key=lambda x: x[1])[0]
                    new_best = max(new_action_probs.items(), key=lambda x: x[1])[0]
                    
                    if old_best != new_best:
                        policy_changed = True
                        changes_count += 1
        
        print(f"   Policy changes: {changes_count} states")
        
        if not policy_changed:
            print("   ✅ No changes - OPTIMAL POLICY FOUND!")
            break
        else:
            print("   ➡️  Policy improved, continuing...")
            current_policy = improved_policy
    
    return current_policy, current_values

# Let's run a quick demo
print("Running Policy Iteration Demo...")
print("This will show you exactly what happens in each iteration!")

# Uncomment the next line to run the demo
# demo_policy, demo_values = simple_policy_iteration_demo(env, gamma=0.9)
```

    Running Policy Iteration Demo...
    This will show you exactly what happens in each iteration!



```python
# Analyze convergence
def plot_value_convergence(V_history):
    """Plot how state values converge over iterations"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Select interesting states to track
    tracked_states = [(0, 0), (1, 0), (2, 2), (3, 2)]
    
    for idx, state in enumerate(tracked_states):
        ax = axes[idx // 2, idx % 2]
        
        values = [V[state] for V in V_history]
        iterations = range(1, len(values) + 1)
        
        ax.plot(iterations, values, 'bo-', linewidth=2, markersize=8)
        ax.set_title(f'Value Convergence for State {state}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Policy Iteration')
        ax.set_ylabel('State Value')
        ax.grid(True, alpha=0.3)
        
        # Annotate final value
        final_value = values[-1]
        ax.annotate(f'Final: {final_value:.3f}', 
                   xy=(len(values), final_value), xytext=(10, 10),
                   textcoords='offset points', fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.suptitle('Convergence Analysis: State Values During Policy Iteration', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.show()

# Plot convergence
plot_value_convergence(V_history)

# Compare policies at different iterations
def compare_policies_over_iterations(env, policy_history, V_history):
    """Compare how the policy changes over iterations"""
    n_policies = min(4, len(policy_history))  # Show up to 4 policies
    
    if n_policies < 2:
        print("Not enough policy iterations to show comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx in range(n_policies):
        ax = axes[idx // 2, idx % 2]
        
        policy = policy_history[idx]
        V = V_history[idx]
        
        # Create policy dictionary for visualization
        policy_dict = {}
        for state in env.states:
            if state != env.goal_state and state not in env.obstacles:
                action_probs = policy.get_action_probs(state)
                if action_probs:
                    best_action = max(action_probs.items(), key=lambda x: x[1])[0]
                    policy_dict[state] = best_action
        
        # Create visualization grid
        grid = np.zeros((env.size, env.size))
        for i, j in env.obstacles:
            grid[i, j] = -1
        goal_i, goal_j = env.goal_state
        grid[goal_i, goal_j] = 1
        
        # Plot
        im = ax.imshow(grid, cmap='RdYlGn', aspect='equal', alpha=0.3)
        
        # Add text annotations
        arrow_map = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
        for i in range(env.size):
            for j in range(env.size):
                state = (i, j)
                if state == env.goal_state:
                    ax.text(j, i, 'G', ha='center', va='center', 
                           fontsize=16, fontweight='bold', color='green')
                elif state in env.obstacles:
                    ax.text(j, i, 'X', ha='center', va='center', 
                           fontsize=16, fontweight='bold', color='red')
                elif state in policy_dict:
                    # Show policy arrow and value
                    action = policy_dict[state]
                    arrow = arrow_map.get(action, '?')
                    ax.text(j, i-0.15, arrow, ha='center', va='center', 
                           fontsize=14, fontweight='bold', color='blue')
                    ax.text(j, i+0.25, f'{V[state]:.2f}', ha='center', va='center', 
                           fontsize=8, color='black')
        
        ax.set_xticks(range(env.size))
        ax.set_yticks(range(env.size))
        ax.set_title(f'Iteration {idx + 1}\n(Policy + Values)', fontsize=12, fontweight='bold')
    
    # Hide empty subplots
    for idx in range(n_policies, 4):
        axes[idx // 2, idx % 2].axis('off')
    
    plt.tight_layout()
    plt.suptitle('Policy Evolution During Policy Iteration', fontsize=16, fontweight='bold')
    plt.show()

compare_policies_over_iterations(env, policy_history, V_history)
```


    
![png](CA2_files/CA2_22_0.png)
    



    
![png](CA2_files/CA2_22_1.png)
    


#
# Part 5: Experiments and Analysis
#
#
# Exercise 5.1: Effect of Discount Factor (γ)**definition:**the Discount Factor Γ Determines How Much We Value Future Rewards Compared to Immediate Rewards.**mathematical Impact:**$$g*t = R*{T+1} + ΓR*{T+2} + Γ^2R*{T+3} + ... = \SUM*{K=0}^{\INFTY} Γ^k R*{T+K+1}$$**INTERPRETATION of Different Values:**- **Γ = 0**: Only Immediate Rewards Matter (myopic Behavior)- **Γ = 1**: All Future Rewards Equally Important (infinite Horizon)- **0 < Γ < 1**: Future Rewards Are Discounted (realistic)**task:** Experiment with Different Discount Factors and Analyze Their Effect on the Optimal Policy.**research QUESTIONS:**1. How Does Γ Affect the Optimal POLICY?2. Which Γ Values Lead to Faster CONVERGENCE?3. What Happens to State Values as Γ Changes?

#
#
# 💰 Discount Factor Deep Dive: Balancing Present Vs Future**core Concept:** the Discount Factor Γ Controls the Agent's "patience" or Time Preference.---
#
#
# ⏰ Time Value of Rewards**financial Analogy:**just like Money, Rewards Have "time Value":- $100 Today Vs $100 in 10 Years → Most Prefer Today (inflation, Uncertainty)- +10 Reward Now Vs +10 Reward in 100 Time Steps → Usually Prefer Immediate**mathematical Effect:**- **Γ = 0.1**: Reward 10 Steps Away Is Worth 0.1¹⁰ = 0.0000000001 of Current Reward- **Γ = 0.9**: Reward 10 Steps Away Is Worth 0.9¹⁰ = 0.35 of Current Reward- **Γ = 0.99**: Reward 10 Steps Away Is Worth 0.99¹⁰ = 0.90 of Current Reward---
#
#
# 🌎 Real-world Analogies**γ = 0.1 (very Impatient/myopic):**- 🍕 "I Want Pizza Now, Don't Care About Health Consequences"- 💳 "BUY with Credit Card, Ignore Interest Charges"- 🚗 "take Fastest Route, Ignore Traffic Fines"**γ = 0.5 (moderately Patient):**- 🏃 "exercise Sometimes for Health Benefits"- 💰 "save Some Money, Spend Some Now"- 📚 "study When Motivated, Party When Not"**γ = 0.9 (balanced):**- 💪 "exercise Regularly for Long-term Health"- 🎓 "study Hard Now for Career Benefits Later"- 💰 "invest Consistently for Retirement"**γ = 0.99 (very Patient):**- 🌱 "plant Trees for Future Generations"- 🏠 "BUY House as Long-term Investment"- 🌍 "address Climate Change for Distant Future"---
#
#
# 📊 Effect on Optimal Policy**low Γ (myopic Behavior):**- Takes Shortest Immediate Path to Reward- Ignores Long-term Consequences- May Get Stuck in Local Optima- Fast Convergence but Potentially Poor Solutions**high Γ (farsighted Behavior):**- Considers Long-term Consequences- May Take Longer Paths for Better Future Outcomes- Explores More Thoroughly- Slower Convergence but Better Final Solutions**in Gridworld Context:**- **low Γ**: Rushes toward Goal, Ignoring Obstacles- **high Γ**: Carefully Plans Path, Avoids Risky Moves
#
#
#
# **mathematical Impact**return Formula**: G*t = R*{T+1} + ΓR*{T+2} + Γ²R*{T+3} + Γ³R*{T+4} + ...**examples**:**γ = 0.9** (patient Agent):- G*t = R*{T+1} + 0.9×R*{T+2} + 0.81×R*{T+3} + 0.729×R*{T+4} + ...- Reward in 1 Step: Worth 100% of Immediate Reward- Reward in 2 Steps: Worth 90% of Immediate Reward - Reward in 3 Steps: Worth 81% of Immediate Reward- Reward in 10 Steps: Worth 35% of Immediate Reward**γ = 0.1** (impatient Agent):- G*t = R*{T+1} + 0.1×R*{T+2} + 0.01×R*{T+3} + 0.001×R_{T+4} + ...- Reward in 2 Steps: Worth Only 10% of Immediate Reward- Reward in 3 Steps: Worth Only 1% of Immediate Reward- Very Myopic - Only Cares About Next Few Steps
#
#
#
# **real-world Analogies**γ = 0.1** (very Impatient):- 🍕 "I Want Pizza Now, Don't Care About Health Consequences"- 📱 "BUY the Cheapest Phone, Ignore Long-term Durability" - 🚗 "take the Fastest Route, Ignore Traffic Fines"**γ = 0.9** (balanced):- 💪 "exercise Now for Health Benefits Later"- 🎓 "study Hard Now for Career Benefits Later"- 💰 "invest Money for Retirement"**γ = 0.99** (very Patient):- 🌱 "plant Trees for Future Generations"- 🏠 "BUY a House as Long-term Investment"- 🌍 "address Climate Change for Distant Future"
#
#
#
# **effect on Optimal Policy**low Γ (myopic Behavior)**:- Takes Shortest Path to Goal- Ignores Long-term Consequences - Might Take Dangerous Shortcuts- Policy: "rush to Goal, Avoid Obstacles Minimally"**high Γ (farsighted Behavior)**:- Takes Safer, Longer Paths- Values Long-term Safety- More Conservative Decisions- Policy: "GET to Goal Safely, Even If It Takes Longer"
#
#
#
# **choosing Γ in PRACTICE**CONSIDER**:1. **problem Horizon**: Short-term Tasks → Lower Γ, Long-term Tasks → Higher Γ2. **uncertainty**: More Uncertain Future → Lower Γ3. **safety**: Safety-critical Applications → Higher Γ4. **computational**: Higher Γ → Slower Convergence**common Values**:- **Γ = 0.9**: General Purpose, Good Balance- **Γ = 0.95-0.99**: Long-term Planning Tasks- **Γ = 0.1-0.5**: Short-term Reactive Tasks- **Γ = 1.0**: Infinite Horizon, Theoretical Studies (CAN Cause Issues)
#
#
#
# **debugging with Γ**if Your Agent:- **ignores Long-term Rewards**: Increase Γ- **IS Too Conservative**: Decrease Γ - **won't Converge**: Check If Γ Is Too Close to 1- **makes Random Decisions**: Γ Might Be Too Low


```python
# Experiment with different discount factors
gamma_values = [0.1, 0.5, 0.9, 0.99]

print("Experimenting with Different Discount Factors:")
print("=" * 60)

results = {}
for gamma in gamma_values:
    print(f"\nγ = {gamma}")
    print("-" * 20)
    
    # Run policy iteration
    opt_policy, V_hist, _ = policy_iteration(env, RandomPolicy(env), gamma=gamma, max_iterations=20)
    
    # Store results
    results[gamma] = {
        'policy': opt_policy,
        'values': V_hist[-1],
        'iterations': len(V_hist)
    }
    
    print(f"Converged in {len(V_hist)} iterations")

# Visualize policies for different gamma values
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, gamma in enumerate(gamma_values):
    ax = axes[idx // 2, idx % 2]
    
    policy = results[gamma]['policy']
    V = results[gamma]['values']
    
    # Create policy visualization
    grid = np.zeros((env.size, env.size))
    for i, j in env.obstacles:
        grid[i, j] = -1
    goal_i, goal_j = env.goal_state
    grid[goal_i, goal_j] = 1
    
    # Add values
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if state not in env.obstacles and state != env.goal_state:
                grid[i, j] = V[state]
    
    im = ax.imshow(grid, cmap='RdYlGn', aspect='equal')
    
    # Add arrows and values
    arrow_map = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    for i in range(env.size):
        for j in range(env.size):
            state = (i, j)
            if state == env.goal_state:
                ax.text(j, i, 'G', ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='darkgreen')
            elif state in env.obstacles:
                ax.text(j, i, 'X', ha='center', va='center', 
                       fontsize=16, fontweight='bold', color='darkred')
            else:
                # Show optimal action
                action_probs = policy.get_action_probs(state)
                if action_probs:
                    best_action = max(action_probs.items(), key=lambda x: x[1])[0]
                    arrow = arrow_map.get(best_action, '?')
                    ax.text(j, i-0.2, arrow, ha='center', va='center', 
                           fontsize=12, fontweight='bold', color='blue')
                # Show value
                ax.text(j, i+0.2, f'{V[state]:.1f}', ha='center', va='center', 
                       fontsize=8, color='black')
    
    ax.set_title(f'γ = {gamma}\n({results[gamma]["iterations"]} iterations)', 
                fontsize=12, fontweight='bold')
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.suptitle('Effect of Discount Factor on Optimal Policy', fontsize=16, fontweight='bold')
plt.show()

# Analyze the effect of gamma on specific states
print("\nValue Analysis for Different Gamma Values:")
print("State\t\t" + "\t".join([f"γ={g}" for g in gamma_values]))
print("-" * 60)

sample_states = [(0,0), (1,0), (2,0), (0,1), (2,2)]
for state in sample_states:
    values_str = "\t".join([f"{results[g]['values'][state]:.2f}" for g in gamma_values])
    print(f"{state}\t\t{values_str}")

print("\nObservations:")
print("- Lower γ → More myopic behavior (focus on immediate rewards)")
print("- Higher γ → More farsighted behavior (plan for long-term rewards)")
print("- γ affects convergence speed and final values")
```

    Experimenting with Different Discount Factors:
    ============================================================
    
    γ = 0.1
    --------------------
    Starting Policy Iteration...
    ==================================================
    
    Iteration 1:
      - Policy Evaluation...
    Policy evaluation converged after 9 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 2:
      - Policy Evaluation...
    Policy evaluation converged after 9 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 3:
      - Policy Evaluation...
    Policy evaluation converged after 9 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 4:
      - Policy Evaluation...
    Policy evaluation converged after 7 iterations
      - Policy Improvement...
    
      ✓ Policy converged after 4 iterations!
    Converged in 4 iterations
    
    γ = 0.5
    --------------------
    Starting Policy Iteration...
    ==================================================
    
    Iteration 1:
      - Policy Evaluation...
    Policy evaluation converged after 23 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 2:
      - Policy Evaluation...
    Policy evaluation converged after 25 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 3:
      - Policy Evaluation...
    Policy evaluation converged after 25 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 4:
      - Policy Evaluation...
    Policy evaluation converged after 7 iterations
      - Policy Improvement...
    
      ✓ Policy converged after 4 iterations!
    Converged in 4 iterations
    
    γ = 0.9
    --------------------
    Starting Policy Iteration...
    ==================================================
    
    Iteration 1:
      - Policy Evaluation...
    Policy evaluation converged after 67 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 2:
      - Policy Evaluation...
    Policy evaluation converged after 154 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 3:
      - Policy Evaluation...
    Policy evaluation converged after 31 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 4:
      - Policy Evaluation...
    Policy evaluation converged after 7 iterations
      - Policy Improvement...
    
      ✓ Policy converged after 4 iterations!
    Converged in 4 iterations
    
    γ = 0.99
    --------------------
    Starting Policy Iteration...
    ==================================================
    
    Iteration 1:
      - Policy Evaluation...
    Policy evaluation converged after 101 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 2:
      - Policy Evaluation...
    Policy evaluation converged after 7 iterations
      - Policy Improvement...
    
      ✓ Policy converged after 2 iterations!
    Converged in 2 iterations



    
![png](CA2_files/CA2_25_1.png)
    


    
    Value Analysis for Different Gamma Values:
    State		γ=0.1	γ=0.5	γ=0.9	γ=0.99
    ------------------------------------------------------------
    (0, 0)		-0.11	0.12	5.50	9.02
    (1, 0)		-0.11	0.44	6.22	9.21
    (2, 0)		-0.10	1.07	7.02	9.41
    (0, 1)		-0.11	0.44	6.22	9.21
    (2, 2)		0.90	4.90	8.90	9.80
    
    Observations:
    - Lower γ → More myopic behavior (focus on immediate rewards)
    - Higher γ → More farsighted behavior (plan for long-term rewards)
    - γ affects convergence speed and final values


#
#
# Exercise 5.2: Modified Environment Experiments**task A**: Modify the Reward Structure and Analyze How It Affects the Optimal Policy:- Change Step Reward from -0.1 to -1.0 (higher Cost for Each Step)- Change Goal Reward from 10 to 5- Add Positive Rewards for Certain States**task B**: Experiment with Different Obstacle Configurations:- Remove Some Obstacles- Add More Obstacles- Change Obstacle Positions**task C**: Test with Different Starting Positions and Analyze Convergence.


```python
# Experiment A - Modified Reward Structure
print("Experiment A: Modified Reward Structure")
print("=" * 40)

# Create environment with different rewards
env_modified = GridWorld(size=4, goal_reward=5, step_reward=-1.0, obstacle_reward=-10)

print("Original vs Modified Environment:")
print(f"Original: goal={env.goal_reward}, step={env.step_reward}, obstacle={env.obstacle_reward}")
print(f"Modified: goal={env_modified.goal_reward}, step={env_modified.step_reward}, obstacle={env_modified.obstacle_reward}")

# Run policy iteration on modified environment
modified_policy, modified_V_hist, _ = policy_iteration(env_modified, RandomPolicy(env_modified), gamma=0.9)

# Compare with original results
print("\nModified Environment Results:")
print(f"Converged in {len(modified_V_hist)} iterations")

# Get final values and policy from modified environment
modified_V_final = modified_V_hist[-1]
modified_policy_dict = {}
for state in env_modified.states:
    if state != env_modified.goal_state and state not in env_modified.obstacles:
        action_probs = modified_policy.get_action_probs(state)
        if action_probs:
            best_action = max(action_probs.items(), key=lambda x: x[1])[0]
            modified_policy_dict[state] = best_action

# Visualize original environment
print("\nOriginal Environment:")
env.visualize_grid(values=V_greedy, title="Original Environment - Optimal Policy")

# Visualize modified environment
print("\nModified Environment:")
env_modified.visualize_grid(values=modified_V_final, policy=modified_policy_dict,
                           title="Modified Environment - Optimal Policy")

# Compare state values
print("\nValue Comparison (sample states):")
print("State\t\tOriginal\tModified\tDifference")
print("-" * 50)
sample_states = [(0,0), (1,0), (2,0), (2,2)]
for state in sample_states:
    orig_val = V_greedy.get(state, 0)
    mod_val = modified_V_final.get(state, 0)
    diff = mod_val - orig_val
    print(f"{state}\t\t{orig_val:.2f}\t\t{mod_val:.2f}\t\t{diff:.2f}")

print("\nObservations:")
print("- Higher step cost (-1.0 vs -0.1) makes agent more eager to reach goal quickly")
print("- Lower goal reward (5 vs 10) reduces overall state values")
print("- Higher obstacle penalty (-10 vs -5) makes agent more cautious around obstacles")

print("\n" + "="*60)
print("Experiment B: Different Obstacle Configurations")
print("=" * 60)

# Create environment with fewer obstacles
env_fewer_obstacles = GridWorld(size=4, goal_reward=10, step_reward=-0.1, obstacle_reward=-5)
env_fewer_obstacles.obstacles = [(1,1)]  # Only one obstacle instead of three
env_fewer_obstacles._build_transition_model()

# Run policy iteration
policy_fewer_obs, V_hist_fewer, _ = policy_iteration(env_fewer_obstacles, 
                                                    RandomPolicy(env_fewer_obstacles), gamma=0.9)

print(f"Fewer obstacles environment converged in {len(V_hist_fewer)} iterations")

# Create environment with different obstacle positions
env_diff_obstacles = GridWorld(size=4, goal_reward=10, step_reward=-0.1, obstacle_reward=-5)
env_diff_obstacles.obstacles = [(0,2), (2,0), (3,1)]  # Different positions
env_diff_obstacles._build_transition_model()

# Run policy iteration
policy_diff_obs, V_hist_diff, _ = policy_iteration(env_diff_obstacles, 
                                                  RandomPolicy(env_diff_obstacles), gamma=0.9)

print(f"Different obstacles environment converged in {len(V_hist_diff)} iterations")

# Visualize all three obstacle configurations
print("\nOriginal Obstacles Configuration:")
env.visualize_grid(values=V_greedy, title="Original Obstacles: (1,1), (2,1), (1,2)")

print("\nFewer Obstacles Configuration:")
V_final_fewer = V_hist_fewer[-1]
policy_dict_fewer = {}
for state in env_fewer_obstacles.states:
    if state != env_fewer_obstacles.goal_state and state not in env_fewer_obstacles.obstacles:
        action_probs = policy_fewer_obs.get_action_probs(state)
        if action_probs:
            best_action = max(action_probs.items(), key=lambda x: x[1])[0]
            policy_dict_fewer[state] = best_action

env_fewer_obstacles.visualize_grid(values=V_final_fewer, policy=policy_dict_fewer,
                                  title="Fewer Obstacles: (1,1)")

print("\nDifferent Obstacles Configuration:")
V_final_diff = V_hist_diff[-1]
policy_dict_diff = {}
for state in env_diff_obstacles.states:
    if state != env_diff_obstacles.goal_state and state not in env_diff_obstacles.obstacles:
        action_probs = policy_diff_obs.get_action_probs(state)
        if action_probs:
            best_action = max(action_probs.items(), key=lambda x: x[1])[0]
            policy_dict_diff[state] = best_action

env_diff_obstacles.visualize_grid(values=V_final_diff, policy=policy_dict_diff,
                                 title="Different Obstacles: (0,2), (2,0), (3,1)")

print("\nObservation Summary:")
print("- Fewer obstacles: Higher overall values, more direct paths available")
print("- Different positions: Changes optimal policy significantly")
print("- Obstacle placement critically affects navigation strategies")

print("\n" + "="*60)

# Template for creating different environments
def create_custom_environment(size=4, goal_reward=10, step_reward=-0.1, 
                             obstacle_reward=-5, obstacles=None):
    """Create a custom GridWorld environment"""
    env = GridWorld(size, goal_reward, step_reward, obstacle_reward)
    if obstacles:
        env.obstacles = obstacles
        env._build_transition_model()  # Rebuild transition model
    return env

# Example: Environment with fewer obstacles
# env_few_obstacles = create_custom_environment(obstacles=[(1, 1)])
# TODO: Test this environment

print("Use the create_custom_environment function to create your own environments!")
```

    Experiment A: Modified Reward Structure
    ========================================
    Original vs Modified Environment:
    Original: goal=10, step=-0.1, obstacle=-5
    Modified: goal=5, step=-1.0, obstacle=-10
    Starting Policy Iteration...
    ==================================================
    
    Iteration 1:
      - Policy Evaluation...
    Policy evaluation converged after 72 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 2:
      - Policy Evaluation...
    Policy evaluation converged after 7 iterations
      - Policy Improvement...
    
      ✓ Policy converged after 2 iterations!
    
    Modified Environment Results:
    Converged in 2 iterations
    
    Original Environment:



    
![png](CA2_files/CA2_27_1.png)
    


    
    Modified Environment:



    
![png](CA2_files/CA2_27_3.png)
    


    
    Value Comparison (sample states):
    State		Original	Modified	Difference
    --------------------------------------------------
    (0, 0)		5.50		-1.14		-6.64
    (1, 0)		6.22		-0.16		-6.38
    (2, 0)		7.02		0.94		-6.08
    (2, 2)		8.90		3.50		-5.40
    
    Observations:
    - Higher step cost (-1.0 vs -0.1) makes agent more eager to reach goal quickly
    - Lower goal reward (5 vs 10) reduces overall state values
    - Higher obstacle penalty (-10 vs -5) makes agent more cautious around obstacles
    
    ============================================================
    Experiment B: Different Obstacle Configurations
    ============================================================
    Starting Policy Iteration...
    ==================================================
    
    Iteration 1:
      - Policy Evaluation...
    Policy evaluation converged after 86 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 2:
      - Policy Evaluation...
    Policy evaluation converged after 154 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 3:
      - Policy Evaluation...
    Policy evaluation converged after 7 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 2:
      - Policy Evaluation...
    Policy evaluation converged after 154 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 3:
      - Policy Evaluation...
    Policy evaluation converged after 7 iterations
      - Policy Improvement...
    
      ✓ Policy converged after 3 iterations!
    Fewer obstacles environment converged in 3 iterations
    Starting Policy Iteration...
    ==================================================
    
    Iteration 1:
      - Policy Evaluation...
    Policy evaluation converged after 68 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 2:
      - Policy Evaluation...
    
      ✓ Policy converged after 3 iterations!
    Fewer obstacles environment converged in 3 iterations
    Starting Policy Iteration...
    ==================================================
    
    Iteration 1:
      - Policy Evaluation...
    Policy evaluation converged after 68 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 2:
      - Policy Evaluation...
    Policy evaluation converged after 154 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 3:
      - Policy Evaluation...
    Policy evaluation converged after 154 iterations
      - Policy Improvement...
      → Policy changed, continuing...
    
    Iteration 3:
      - Policy Evaluation...
    Policy evaluation converged after 154 iterations
      - Policy Improvement...
    
      ✓ Policy converged after 3 iterations!
    Different obstacles environment converged in 3 iterations
    
    Original Obstacles Configuration:
    Policy evaluation converged after 154 iterations
      - Policy Improvement...
    
      ✓ Policy converged after 3 iterations!
    Different obstacles environment converged in 3 iterations
    
    Original Obstacles Configuration:



    
![png](CA2_files/CA2_27_5.png)
    


    
    Fewer Obstacles Configuration:



    
![png](CA2_files/CA2_27_7.png)
    


    
    Different Obstacles Configuration:



    
![png](CA2_files/CA2_27_9.png)
    


    
    Observation Summary:
    - Fewer obstacles: Higher overall values, more direct paths available
    - Different positions: Changes optimal policy significantly
    - Obstacle placement critically affects navigation strategies
    
    ============================================================
    Use the create_custom_environment function to create your own environments!


#
# Part 6: Summary and Key Takeaways
#
#
# What We've LEARNED**1. Markov Decision Processes (mdps):**- **framework**: Sequential Decision Making under Uncertainty- **components**: (S, A, P, R, Γ) - States, Actions, Transitions, Rewards, Discount- **markov Property**: Future Depends Only on Current State, Not History- **foundation**: Mathematical Basis for All Rl ALGORITHMS**2. Value Functions:**- **v^π(s)**: Expected Return Starting from State S Following Policy Π - **q^π(s,a)**: Expected Return Taking Action a in State S, Then Following Π- **relationship**: V^π(s) = Σ*a Π(a|s) Q^π(s,a)- **purpose**: Measure "goodness" of States and ACTIONS**3. Bellman Equations:**- **for V**: V^π(s) = Σ*a Π(a|s) Σ*{s'} P(s'|s,a)[r(s,a,s') + Γv^π(s')]- **for Q**: Q^π(s,a) = Σ*{s'} P(s'|s,a)[r(s,a,s') + Γ Σ*{a'} Π(a'|s')q^π(s',a')]- **significance**: Recursive Relationship Enabling Dynamic Programming SOLUTIONS**4. Policy Evaluation:**- **algorithm**: Iterative Method to Compute V^π Given Policy Π- **convergence**: Guaranteed for Finite Mdps with Γ < 1- **application**: Foundation for Policy Iteration and Value ITERATION**5. Policy Improvement:**- **theorem**: Greedy Policy W.r.t. V^π Is at Least as Good as Π- **formula**: Π'(s) = Argmax*a Q^π(s,a)- **monotonicity**: Each Improvement Step Yields Better or Equal POLICY**6. Policy Iteration:**- **algorithm**: Alternates between Evaluation and Improvement- **guarantee**: Converges to Optimal Policy Π*- **efficiency**: Usually Converges in Few Iterations---
#
#
# Key Insights from Experiments**discount Factor (Γ) Effects:**- **low Γ**: Myopic Behavior, Focuses on Immediate Rewards- **high Γ**: Farsighted Behavior, Considers Long-term Consequences- **trade-off**: Convergence Speed Vs Solution Quality**environment Structure Impact:**- **reward Structure**: Significantly Affects Optimal Policy- **obstacles**: Create Navigation Challenges Requiring Planning- **starting Position**: Can Influence Learning Dynamics**algorithm Characteristics:**- **model-based**: Requires Knowledge of P(s'|s,a) and R(s,a,s')- **exact Solution**: Finds Truly Optimal Policy (unlike Approximate Methods)- **computational Cost**: Scales with State Space Size---
#
#
# Connections to Advanced Topics**what This Enables:**- **value Iteration**: Direct Optimization of Value Function- **q-learning**: Model-free Learning of Action-value Functions- **deep Rl**: Neural Network Function Approximation- **policy Gradients**: Direct Policy Optimization Methods**next Steps in Learning:**- **temporal Difference Learning**: Learn from Incomplete Episodes- **function Approximation**: Handle Large/continuous State Spaces- **exploration Vs Exploitation**: Balance Learning and Performance- **multi-agent Systems**: Multiple Learning Agents Interacting---
#
#
# Reflection Questions**theoretical UNDERSTANDING:**1. How Would Stochastic Transitions Affect the Optimal POLICY?2. What Happens with Continuous State or Action SPACES?3. How Do We Handle Unknown Environment DYNAMICS?4. What Are Computational Limits for Large State Spaces?**practical APPLICATIONS:**1. How Could You Apply Mdps to Real-world Decision PROBLEMS?2. What Modifications Would Be Needed for Competitive SCENARIOS?3. How Would You Handle Partially Observable ENVIRONMENTS?4. What Safety Considerations Are Important in Rl Applications?

#
#
# 🧠 Common Misconceptions and Intuitive Understandingbefore We Wrap Up, Let's Address Some Common Confusions and Solidify Understanding:
#
#
#
# **❌ Common MISCONCEPTIONS**1. "value Functions Are Just Rewards"**- ❌ Wrong: V(s) ≠ R(s) - ✅ Correct: V(s) = Expected Total Future Reward from State S- 🔍 Think: V(s) Is like Your Bank Account Balance, R(s) Is Your Daily INCOME**2. "q(s,a) Tells Me the Best Action"**- ❌ Wrong: Q(s,a) Is Not Binary Good/bad- ✅ Correct: Q(s,a) Is the Expected Value of Taking Action A- 🔍 Think: Compare Q-values to Choose Best Action: Argmax_a Q(S,A)**3. "policy Iteration Always Takes Many Steps"**- ❌ Wrong: Often Converges in 2-4 Iterations- ✅ Correct: Convergence Is Usually Very Fast- 🔍 Think: Once You Find a Good Strategy, Small Improvements Are ENOUGH**4. "random Policy Is Always Bad"**- ❌ Wrong: Random Policy Can Be Good for Exploration- ✅ Correct: Depends on Environment and Goals- 🔍 Think: Sometimes Trying New Things Leads to Better Discoveries
#
#
#
# **🎯 Key Intuitions to REMEMBER**1. the Big Picture Flow**:```environment → Policy → Actions → Rewards → Better Policy → REPEAT```**2. Value Functions as Gps**:- V(s): "HOW Good Is This Location Overall?"- Q(s,a): "HOW Good Is Taking This Road from This LOCATION?"**3. Bellman Equations as Consistency**:- "MY Value Should Equal Immediate Reward + Discounted Future Value"- Like: "MY Wealth = Today's Income + Tomorrow's WEALTH"**4. Policy Improvement as Learning**:- "IF I Know What Each Action Leads To, I Can Choose Better Actions"- Like: "IF I Know Exam Results for Each Study Method, I Can Study Better"
#
#
#
# **🔧 Troubleshooting Guide****if Values Don't Converge**:- Check If Γ < 1 - Reduce Convergence Threshold (theta)- Check for Bugs in Transition Probabilities**if Policy Doesn't Improve**:- Environment Might Be Too Simple (already Optimal)- Check Reward Structure - Might Need More Differentiation- Verify Policy Improvement Logic**if Results Seem Weird**:- Visualize Value Functions and Policies- Start with Simpler Environment- Check Reward Signs (positive/negative)
#
#
#
# **🚀 Connecting to Future Topics**what We Learned Here Enables:- **value Iteration**: Direct Value Optimization (next Week!)- **q-learning**: Learn Q-values without Knowing the Model- **deep Rl**: Use Neural Networks to Handle Large State Spaces- **policy Gradients**: Directly Optimize the Policy Parameters
#
#
#
# **🎭 the Rl Mindset**think like an Rl AGENT:1. **observe** Your Current Situation (STATE)2. **consider** Your Options (actions) 3. **predict** Outcomes (USE Your MODEL/EXPERIENCE)4. **choose** the Best Option (POLICY)5. **learn** from Results (update VALUES/POLICY)6. **repeat** until Masterythis Mindset Applies To:- Career Decisions- Investment Choices - Game Strategies- Daily Life Optimization


```python
# Final Code Cell - Additional Experiments and Testing

print("🎯 Congratulations! You've completed the Deep Reinforcement Learning Session 2 Exercise")
print("=" * 80)

print("\n📋 Summary of Implemented Components:")
print("✓ GridWorld Environment")
print("✓ Policy Classes (Random, Greedy, Custom)")
print("✓ Policy Evaluation Algorithm")
print("✓ Q-value Computation")
print("✓ Policy Improvement")
print("✓ Policy Iteration Algorithm")
print("✓ Convergence Analysis")
print("✓ Discount Factor Experiments")

print("\n🧪 Optional Challenges:")
print("1. Implement Value Iteration algorithm")
print("2. Add stochastic transitions (wind, slippery grid)")
print("3. Create larger grid worlds (8x8, 10x10)")
print("4. Implement different policy representation (soft-max, epsilon-greedy)")
print("5. Add time-dependent rewards")

print("\n💡 Use this space for your additional experiments:")
print("# Your custom experiments go here...")

# Example: Quick test of your understanding
def quick_test():
    """Quick test to verify understanding"""
    print("\n🔍 Quick Understanding Test:")
    
    # Create a simple 2x2 grid for testing
    test_env = GridWorld(size=2, goal_reward=1, step_reward=0, obstacle_reward=0)
    test_env.obstacles = []  # Remove obstacles for simplicity
    test_env._build_transition_model()
    
    # Test policy evaluation
    random_pol = RandomPolicy(test_env)
    V_test = policy_evaluation(test_env, random_pol, gamma=0.9, theta=1e-10)
    
    print(f"Simple 2x2 grid values: {V_test}")
    print("Expected: Values should decrease as we move away from goal (1,1)")
    
    return V_test

# Run quick test
# quick_test()

print("\n🚀 Ready for the next session: Temporal Difference Learning!")
print("Keep experimenting and happy learning! 🤖")
```

    🎯 Congratulations! You've completed the Deep Reinforcement Learning Session 2 Exercise
    ================================================================================
    
    📋 Summary of Implemented Components:
    ✓ GridWorld Environment
    ✓ Policy Classes (Random, Greedy, Custom)
    ✓ Policy Evaluation Algorithm
    ✓ Q-value Computation
    ✓ Policy Improvement
    ✓ Policy Iteration Algorithm
    ✓ Convergence Analysis
    ✓ Discount Factor Experiments
    
    🧪 Optional Challenges:
    1. Implement Value Iteration algorithm
    2. Add stochastic transitions (wind, slippery grid)
    3. Create larger grid worlds (8x8, 10x10)
    4. Implement different policy representation (soft-max, epsilon-greedy)
    5. Add time-dependent rewards
    
    💡 Use this space for your additional experiments:
    # Your custom experiments go here...
    
    🚀 Ready for the next session: Temporal Difference Learning!
    Keep experimenting and happy learning! 🤖



```python
# 🎯 INTERACTIVE LEARNING EXERCISES
print("=" * 80)
print("🎓 SELF-CHECK QUESTIONS - Test Your Understanding!")
print("=" * 80)

def self_check_questions():
    """
    Interactive questions to test understanding
    Run this function and think about your answers before revealing the solutions
    """
    
    questions = [
        {
            "q": "🤔 Q1: If γ = 0, what does the agent care about?",
            "options": ["A) Only immediate rewards", "B) All future rewards equally", "C) Long-term rewards more"],
            "answer": "A",
            "explanation": "γ=0 means future rewards are multiplied by 0, so only immediate rewards matter!"
        },
        {
            "q": "🤔 Q2: What does V(s) represent?",
            "options": ["A) Immediate reward in state s", "B) Best action in state s", "C) Expected future reward from state s"],
            "answer": "C", 
            "explanation": "V(s) is the expected cumulative reward starting from state s and following the policy."
        },
        {
            "q": "🤔 Q3: How does policy iteration guarantee finding the optimal policy?",
            "options": ["A) Random exploration", "B) Monotonic improvement + finite policies", "C) Magic"],
            "answer": "B",
            "explanation": "Each iteration improves (or keeps same) policy, and there are finite possible policies, so must reach optimum!"
        },
        {
            "q": "🤔 Q4: In our GridWorld, why is the step reward negative?",
            "options": ["A) To punish the agent", "B) To encourage efficiency", "C) Random choice"],
            "answer": "B",
            "explanation": "Negative step reward encourages finding shorter paths - otherwise agent might wander forever!"
        }
    ]
    
    print("Think about each question, then uncomment the reveal_answers() call to see solutions!\n")
    
    for i, item in enumerate(questions, 1):
        print(f"{item['q']}")
        for option in item['options']:
            print(f"   {option}")
        print()
    
    def reveal_answers():
        print("🔍 ANSWERS AND EXPLANATIONS:")
        print("-" * 50)
        for i, item in enumerate(questions, 1):
            print(f"Q{i}: Answer is {item['answer']}")
            print(f"💡 Explanation: {item['explanation']}\n")
    
    return reveal_answers

# Run the self-check
reveal_function = self_check_questions()

# Uncomment the next line to reveal answers:
# reveal_function()

print("\n" + "="*80)
print("🏆 HANDS-ON CHALLENGES - Try These!")
print("="*80)

print("""
🚀 CHALLENGE 1: Modify the Environment
   - Create a 6x6 grid with different obstacle patterns
   - Try diagonal obstacles, maze-like structures
   - Compare how policies change

🚀 CHALLENGE 2: Custom Reward Structure  
   - Add "bonus" states with positive rewards
   - Create "danger" zones with high negative rewards
   - See how this affects optimal paths

🚀 CHALLENGE 3: Stochastic Environment
   - Modify transitions to be probabilistic
   - E.g., 80% chance of intended direction, 20% random
   - Compare convergence and policies

🚀 CHALLENGE 4: Analyze Different Starting Points
   - What if agent starts at different corners?
   - How does this change the value function?
   - Create heat maps for different starting positions

🚀 CHALLENGE 5: Policy Comparison
   - Implement epsilon-greedy policy (90% greedy, 10% random)  
   - Compare convergence speed vs quality
   - Which performs better in different scenarios?
""")

print("💡 DEBUGGING TIPS:")
print("-" * 30)
print("✓ Always visualize your results")
print("✓ Start with simple cases (2x2 grid)")
print("✓ Check that probabilities sum to 1")
print("✓ Verify rewards make intuitive sense")
print("✓ Use different γ values to understand behavior")

print(f"\n🎉 You've completed the comprehensive DRL Session 2 exercise!")
print(f"🌟 You now understand: MDPs, Policies, Value Functions, Bellman Equations, and Policy Iteration!")
print(f"🚀 Ready for more advanced topics like Q-learning and Deep RL!")

# Fun fact generator
import random
fun_facts = [
    "🤖 The Bellman equation is named after Richard Bellman, who also invented dynamic programming!",
    "🎮 Many modern game AIs use variations of the algorithms you just learned!",
    "🚗 Tesla's autopilot uses deep reinforcement learning for decision making!",
    "🏆 AlphaGo used policy iteration concepts to master the game of Go!",
    "💰 Wall Street uses RL algorithms for algorithmic trading!",
    "🎯 The epsilon-greedy strategy balances exploration vs exploitation - key in RL!"
]

print(f"\n🎲 Fun RL Fact: {random.choice(fun_facts)}")
print("\n" + "="*80)
```

    ================================================================================
    🎓 SELF-CHECK QUESTIONS - Test Your Understanding!
    ================================================================================
    Think about each question, then uncomment the reveal_answers() call to see solutions!
    
    🤔 Q1: If γ = 0, what does the agent care about?
       A) Only immediate rewards
       B) All future rewards equally
       C) Long-term rewards more
    
    🤔 Q2: What does V(s) represent?
       A) Immediate reward in state s
       B) Best action in state s
       C) Expected future reward from state s
    
    🤔 Q3: How does policy iteration guarantee finding the optimal policy?
       A) Random exploration
       B) Monotonic improvement + finite policies
       C) Magic
    
    🤔 Q4: In our GridWorld, why is the step reward negative?
       A) To punish the agent
       B) To encourage efficiency
       C) Random choice
    
    
    ================================================================================
    🏆 HANDS-ON CHALLENGES - Try These!
    ================================================================================
    
    🚀 CHALLENGE 1: Modify the Environment
       - Create a 6x6 grid with different obstacle patterns
       - Try diagonal obstacles, maze-like structures
       - Compare how policies change
    
    🚀 CHALLENGE 2: Custom Reward Structure  
       - Add "bonus" states with positive rewards
       - Create "danger" zones with high negative rewards
       - See how this affects optimal paths
    
    🚀 CHALLENGE 3: Stochastic Environment
       - Modify transitions to be probabilistic
       - E.g., 80% chance of intended direction, 20% random
       - Compare convergence and policies
    
    🚀 CHALLENGE 4: Analyze Different Starting Points
       - What if agent starts at different corners?
       - How does this change the value function?
       - Create heat maps for different starting positions
    
    🚀 CHALLENGE 5: Policy Comparison
       - Implement epsilon-greedy policy (90% greedy, 10% random)  
       - Compare convergence speed vs quality
       - Which performs better in different scenarios?
    
    💡 DEBUGGING TIPS:
    ------------------------------
    ✓ Always visualize your results
    ✓ Start with simple cases (2x2 grid)
    ✓ Check that probabilities sum to 1
    ✓ Verify rewards make intuitive sense
    ✓ Use different γ values to understand behavior
    
    🎉 You've completed the comprehensive DRL Session 2 exercise!
    🌟 You now understand: MDPs, Policies, Value Functions, Bellman Equations, and Policy Iteration!
    🚀 Ready for more advanced topics like Q-learning and Deep RL!
    
    🎲 Fun RL Fact: 🤖 The Bellman equation is named after Richard Bellman, who also invented dynamic programming!
    
    ================================================================================



```python

```
