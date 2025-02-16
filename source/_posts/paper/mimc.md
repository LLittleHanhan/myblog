
**MiMC** is a block cipher and hash function family designed specifically for SNARK applications. The low multiplicative complexity of MiMC over prime fields makes it suitable for ZK-SNARK applications such as ZCash.

The core component of MiMC is the APN function . The function is computed in \\(\\mathbb{F}\_q\\), where \\(q = p\\) or \\(q = 2^n\\) for a prime number \\(p\\) and a natural number \\(n\\).

The key scheduling adds the same (uniformly randomly chosen secret) key \\(k \\in \\mathbb{F}\_q\\) at each round and is followed by the round constant addition. In detail, the encryption function of MiMC is

where is the plaintext, is the number of rounds, \\(F\_i\\) is the round function for round \\(i \\geq 0\\), and \\(k \\in \\mathbb{F}\_q\\) is the key. Each \\(F\_i\\) is defined as

where \\(c\_i \\in \\mathbb{F}\_q\\) are the round constants and \\(c\_0 = 0\\). The round constants are chosen as random elements of \\(\\mathbb{F}\_q\\) at the instantiation of MiMC and then fixed. Note that there are no round keys, instead the same key is used in each round and once at the end. All the operations are defined in the underlying field .

![](https://byt3bit.github.io/primesym/mimc/mimc.png)

For a -bit key, is chosen uniformly randomly. In this case the two keys and are added alternately through the rounds. Hence, the key scheduling for round is defined as , and the round function is defined as

for .

**Feistel-MiMC** is constructed over using the same round function \\(f(x) = x^3\\). Note that in Feistel-MiMC the input (and output) is in . Each round of is defined as

where is the state after round , is the input and for \\(i \\geq 0 \\).
