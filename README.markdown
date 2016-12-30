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

> q : [donald trump won last nights presidential debate according to snap online polls]; a : [thought he was a joke]
q : [still i think april unk v u would make good radio]; a : [the whole thing is the worst of the game]
q : [angela is proof of why you cant play with bitches who are in their feelings]; a : [lol if you want to be a good one of the world]
q : [chop that shit up while blasting death metal]; a : [but he has a good job]
q : [just wanna live in unk everything is 10x better there]; a : [i was just on the same side and i was like it was a good time]
q : [the lil girl i forgot her name scares the shit out of me n probably always will lmaooo]; a : [she was so cute and she was so cute and she was a bitch]
q : [love this powerful story about how role models change a childs dreams via]; a : [i think they were talking about the same side of the debate]
q : [day unk by  have fun in ac]; a : [congrats to you ]
q : [chelsea clinton unk here that marijuana can kill you uhh]; a : [the only reason is that]
q : [what is everyone reading these days]; a : [i have a feeling that was a good game of the game]
q : [that works wanna do coffee or unk too]; a : [its a good thing]
q : [maybe i missed it but what the heck happened with unk]; a : [it was a good idea]
q : [i dont live in new york because of their loose common law marriage unk]; a : [i think its a threat to the white states and the media movement]
q : [excuse me do you have the time  its ]; a : [i dont know what i was there]
q : [where can i stay posted]; a : [if i can get my phone to get a new phone]

## Credits

Borrowed most of the code from [mikesj-public](https://github.com/mikesj-public/rnn_spelling_bee/blob/master/spelling_bee_RNN.ipynb)
