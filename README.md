## Deck-based divide-the-dollar

A deck-based game is a mathematical game that has been transformed into a card game in which potential moves are limited to a set of cards drawn from a fixed deck <a name="a1"></a>[[1](#AshlockSchonfeldCardGames)]. Each player must choose their strategy during each turn by selecting a card from their hand. Once chosen, the card is removed and cannot be played again.

We examine here a deck-based version of the divide-the-dollar game. Divide-the-dollar is typically a simultaneous play (generalized Nash) game in which two players each make a bid to divide a dollar. If the sum of the bids is less than one, then each player gets to keep the amount of their bid <a name="a2"></a>[[2](#BramsTaylor)]. We will look at an iterated, two-player, turn- and deck-based version of this in which we have multiple rounds of players taking turns playing the first card.

```python
from __future__ import division
import numpy as np
import math
```

### Game parameters

First, we must specify the game itself. This includes defining the specific fractions used as cards as well as the corresponding number of these unique cards that we wish to include in a deck.

```python
cards = [0.25, 0.50, 0.75]
num_unique_cards = [16, 28, 16]
num_cards = len(cards)
deck_size = sum(num_unique_cards)
```

Following this we need to define the number of players, the number of cards each player holds in their hand, and the number of rounds which constitutes a game. In this example, players will play cards until the deck is depleted and they can no longer maintain the specified hand size.

```python
num_players = 2
hand_size = 5
num_rounds = 1+(deck_size-(num_players*hand_size))//num_players
```

### Actions

In order to simplify the decision-making process during each round, we define three actions available to players. These actions include playing their largest card (that maximizes score), playing their median card, and playing their smallest card (that spoils the round if playing second). 

### Game state information

Certain information about the current state of the game is available to players during their turn. This information includes whether they are playing first or second; the values of their smallest, median, and largest cards; and the value of the cards played so far (will be zero if playing first).

|Variable|Description|
|:------:|:----------|
|```total```|total value of cards played this round|
|```small```|value of player's smallest card|
|```med```|value of player's median card|
|```large```|value of player's largest card|
|```index```|0 if playing first, 1 if playing second|

## References

<a name="AshlockSchonfeldCardGames"></a> [[1](#a1)] D. Ashlock and J. Schonfeld. [Tools for deriving card games from mathematical games](http://eldar.mathstat.uoguelph.ca/dashlock/eprints/GTRY16.pdf). *Game & Puzzle Design*, 1(2):1–3, 2015.

<a name="BramsTaylor"></a> [[2](#a2)] S. J. Brams and A. D. Taylor. [Divide the dollar: three solutions and extensions](https://doi.org/10.1007/BF01079266). *Theory and Decision*, 37:211–231, 1994.