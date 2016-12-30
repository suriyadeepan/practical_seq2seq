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
| still i think april unk v u would make good radio | the whole thing is the worst of the game |
| angela is proof of why you cant play with bitches who are in their feelings | lol if you want to be a good one of the world |
| chop that shit up while blasting death metal | but he has a good job |
| just wanna live in unk everything is 10x better there | i was just on the same side and i was like it was a good time |
| the lil girl i forgot her name scares the shit out of me n probably always will lmaooo | she was so cute and she was so cute and she was a bitch |
| love this powerful story about how role models change a childs dreams via | i think they were talking about the same side of the debate |
| day unk by  have fun in ac | congrats to you  |
| chelsea clinton unk here that marijuana can kill you uhh | the only reason is that |
| what is everyone reading these days | i have a feeling that was a good game of the game |
| that works wanna do coffee or unk too | its a good thing |
| maybe i missed it but what the heck happened with unk | it was a good idea |
| i dont live in new york because of their loose common law marriage unk | i think its a threat to the white states and the media movement |
| excuse me do you have the time  its  | i dont know what i was there |
| where can i stay posted | if i can get my phone to get a new phone |

## Credits

- Borrowed most of the code for [seq2seq_wrapper.py](/seq2seq_wrapper.py) from [mikesj-public](https://github.com/mikesj-public/rnn_spelling_bee/blob/master/spelling_bee_RNN.ipynb)
- Borrowed the Twitter dataset from this dude : [Marsan-Ma](https://github.com/Marsan-Ma/)
