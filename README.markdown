# Seq2Seq Wrapper for Tensorflow

*requires tensorflow [0.12.0]*

To make life easier for beginners looking to experiment with seq2seq model.


## Datasets

- [x] [CMU Pronouncing Dictionary](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/CMUdict) : Phoneme sequence to word (sequence of alphabets)
![](/img/phoneme.png)
- [x] [Twitter Chat Log](https://github.com/suriyadeepan/datasets/tree/master/seq2seq/twitter) : Sentence to Sentence

![](/img/twitter01.png)
![](/img/twitter02.png)

**Update**

After tweaking the dataset to bring down the overall % of unknown tokens to 3%, increasing vocabulary to 8000, the [results](https://gist.github.com/suriyadeepan/5d4aaf96c7bd89b908921804c683fee8) are promising.

| Query					| Reply					|
| ------------- | ------------- |
| donald trump won last nights presidential debate according to snap online polls | thought he was a joke |
| trump campaign please never delete this | he was a joke |
| s new scoop is unk the castro connection how trumps company violated the us embargo against cuba | trump is a disgrace to the debate and the media is not a liar |
| who won the first presidential debate | trump will be a better time for a man |
| just wanna live in unk everything is 10x better there | i was just on the same side and i was like it was a good time |
| the lil girl i forgot her name scares the shit out of me n probably always will lmaooo | she was so cute and she was so cute and she was a bitch |
| question of the day who do you think won last nights debate weigh in using | hillary is a trump supporter and a woman who has a better job than a woman |
| trump is a total embarrassment hes like a unk child who happens to be a professional liar and con man | trump is a lot of money to get a great country |
| chill the fuck out | thats what i said |
| so fun to see what the special effects team created for | wow that is the best thing ive ever seen |
| she was fine too unk voice | she is a good guy |
| the important ones are still here | i dont think so |
| been drinking pumpkin spice protein shake every morning for a week and now i literally cannot even | me too i just got it |
| lmao i just lost my job and i cant even leave early to be in the unk video today | literally the same thing |
| hey happy birthday have a nice day | [thank you |

## Credits

- Borrowed most of the code for [seq2seq_wrapper.py](/seq2seq_wrapper.py) from [mikesj-public](https://github.com/mikesj-public/rnn_spelling_bee/blob/master/spelling_bee_RNN.ipynb)
- Borrowed the twitter dataset from this dude [Marsan-Ma](https://github.com/Marsan-Ma/)
