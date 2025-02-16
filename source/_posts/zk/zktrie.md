1. AsHash<T>,将T包装成ashash类型，该类型需要实现可变不可变引用 hashable特征
2. SimpleDb<vec<u8>,vec<u8>>,实现ZktrieDatabase


## Node
```rust
pub struct Node<H: Hashable> {
    // node_type is the type of node in the tree.
    pub node_type: NodeType,
    // child_l is the node hash of the left child of a parent node.
    pub child_left: Option<H>,
    // child_r is the node hash of the right child of a parent node.
    pub child_right: Option<H>,
    // key is the node's key stored in a leaf node.
    pub node_key: H,
    // value_preimage can store at most 256 byte32 as fields (represnted by BIG-ENDIAN integer)
    // and the first 24 can be compressed (each bytes32 consider as 2 fields), in hashing the compressed
    // elemments would be calculated first
    pub value_preimage: Vec<[u8; 32]>,
    // use each bit for indicating the compressed flag for the first 24 fields
    pub compress_flags: u32,
    // nodeHash is the cache of the hash of the node to avoid recalculating
    pub node_hash: Option<H>,
    // valueHash is the cache of the hash of valuePreimage to avoid recalculating, only valid for leaf node
    pub value_hash: Option<H>,
    // KeyPreimage is the original key value that derives the node_key, kept here only for proof
    pub key_preimage: Option<[u8; 32]>,
}

pub fn new_node_from_bytes(b: &[u8]) -> Result<Node<H>, ImplError> {}
pub fn canonical_value(&self) -> Vec<u8> {}
pub fn calc_node_hash(mut self) -> Result<Self, ImplError> {}
```

## ZkTrieImpl
```rust
pub struct ZkTrieImpl<H: Hashable, DB: ZktrieDatabase> {
    db: DB,
    root_hash: H,
    writable: bool,
    max_levels: u32,
    debug: bool,
}
```

## ZkTrie
```rust
pub struct ZkTrie<H: Hashable, DB: ZktrieDatabase> {
    tree: ZkTrieImpl<H, DB>,
}
```


## my impl
```cpp
class Node {
   public:
    int type;
    int idx;
    int domain;
    Node* left = nullptr;
    Node* right = nullptr;
    Node* father = nullptr;
};

// only insert
class Trie {
   public:
    Node* root = nullptr;
    // input len = leaf_node_num
    u8* node_value_preimages;
    u8* node_keys;
    // output len = leaf_node_num
    u8* value_hashs;
    // result len = total_node_num
    u8* lvalue;
    u8* rvalue;
    u8* node_hashs;

    vector<Node*> nodes;
    uint leaf_node_num;
    uint total_node_num;
}
```