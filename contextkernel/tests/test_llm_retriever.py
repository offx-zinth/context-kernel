import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

# Modules to test
from contextkernel.core_logic.llm_retriever import (
    LLMRetrieverConfig,
    HuggingFaceEmbeddingModel,
    StubLTM,
    StubGraphDB,
    LLMRetriever,
    RetrievedItem,
    RetrievalResponse
)

# Mock external libraries that are imported at the module level in llm_retriever.py
# We patch them where they are *used* or *imported* in the llm_retriever module.

@pytest.fixture(autouse=True)
def mock_external_libs(monkeypatch):
    """Mocks all major external libraries used by llm_retriever globally for all tests."""

    # Mock sentence_transformers.SentenceTransformer
    mock_sentence_transformer_instance = MagicMock()
    mock_sentence_transformer_instance.encode.return_value = [0.1, 0.2, 0.3] # Dummy embedding
    mock_sentence_transformer_constructor = MagicMock(return_value=mock_sentence_transformer_instance)
    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.SentenceTransformer", mock_sentence_transformer_constructor)

    # Mock sentence_transformers.CrossEncoder
    mock_cross_encoder_instance = MagicMock()
    mock_cross_encoder_instance.predict.return_value = [0.9, 0.8, 0.7] # Dummy scores
    mock_cross_encoder_constructor = MagicMock(return_value=mock_cross_encoder_instance)
    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.CrossEncoder", mock_cross_encoder_constructor)

    # Mock faiss
    mock_faiss = MagicMock()
    mock_faiss_index_instance = MagicMock()
    mock_faiss_index_instance.ntotal = 0
    mock_faiss_index_instance.search.return_value = (MagicMock(), MagicMock()) # (distances, indices)
    mock_faiss.IndexFlatL2.return_value = mock_faiss_index_instance
    mock_faiss.IndexIDMap2.return_value = mock_faiss_index_instance
    mock_faiss.read_index.return_value = mock_faiss_index_instance
    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.faiss", mock_faiss)

    # Mock networkx
    mock_nx = MagicMock()
    mock_nx_graph_instance = MagicMock()
    mock_nx_graph_instance.number_of_nodes.return_value = 0
    mock_nx_graph_instance.number_of_edges.return_value = 0
    mock_nx_graph_instance.has_node.return_value = False
    mock_nx_graph_instance.nodes = {} # Simplified mock
    mock_nx_graph_instance.edges = MagicMock(return_value=[]) # Simplified mock
    mock_nx.Graph.return_value = mock_nx_graph_instance
    mock_nx.read_gml.return_value = mock_nx_graph_instance
    mock_nx.read_graphml.return_value = mock_nx_graph_instance
    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.nx", mock_nx)

    # Mock whoosh
    mock_whoosh_index = MagicMock()
    mock_whoosh_index.searcher.return_value.__enter__.return_value.search.return_value = [] # No hits by default
    mock_whoosh_schema = MagicMock()

    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.create_in", MagicMock(return_value=mock_whoosh_index))
    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.open_dir", MagicMock(return_value=mock_whoosh_index))
    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.exists_in", MagicMock(return_value=False))
    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.Schema", mock_whoosh_schema)
    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.QueryParser", MagicMock())

    # Mock numpy as it's used by StubLTM and FAISS integration
    mock_np = MagicMock()
    mock_np.array.side_effect = lambda x, dtype: x # Just pass through for basic list to "array"
    mock_np.float32 = "float32" # Placeholder for dtype
    mock_np.reshape.side_effect = lambda arr, shape: arr # Pass through
    mock_np.linalg.norm.return_value = 1.0 # Avoid division by zero
    mock_np.dot.return_value = MagicMock() # Mock dot product result
    mock_np.argsort.return_value = MagicMock()
    monkeypatch.setattr("contextkernel.core_logic.llm_retriever.np", mock_np)

    # Mock os.path.exists for index loading paths
    monkeypatch.setattr("os.path.exists", MagicMock(return_value=False))
    monkeypatch.setattr("os.makedirs", MagicMock())


@pytest.fixture
def default_config():
    """Returns a default LLMRetrieverConfig for tests."""
    return LLMRetrieverConfig(
        embedding_model_name="mock-embed-model",
        cross_encoder_model_name=None, # Disabled by default for simpler tests
        faiss_index_path=None,
        networkx_graph_path=None,
        whoosh_index_dir="test_whoosh_index", # Use a test-specific dir
        keyword_search_enabled=True # Enable for testing _search_keyword
    )

# --- Test HuggingFaceEmbeddingModel ---

class TestHuggingFaceEmbeddingModel:
    def test_hf_init_success(self, mock_external_libs):
        """Tests successful initialization of HuggingFaceEmbeddingModel."""
        # The mock_external_libs fixture already mocks SentenceTransformer constructor
        model = HuggingFaceEmbeddingModel(model_name="test-model", device="cpu")
        assert model.model is not None
        # Check that SentenceTransformer was called with the right params
        mock_external_libs_sentence_transformer = contextkernel.core_logic.llm_retriever.SentenceTransformer
        mock_external_libs_sentence_transformer.assert_called_once_with("test-model", device="cpu")

    def test_hf_init_failure_logs_error_and_model_is_none(self, monkeypatch, caplog):
        """Tests that an error during model loading is logged and self.model is None."""
        mock_sentence_transformer_constructor = MagicMock(side_effect=Exception("Model loading failed!"))
        monkeypatch.setattr("contextkernel.core_logic.llm_retriever.SentenceTransformer", mock_sentence_transformer_constructor)

        model = HuggingFaceEmbeddingModel(model_name="bad-model")
        assert model.model is None
        assert "Failed to load SentenceTransformer model 'bad-model': Model loading failed!" in caplog.text

    @pytest.mark.asyncio
    async def test_hf_generate_embedding_model_none(self, caplog):
        """Tests generate_embedding returns [] if model is None."""
        model = HuggingFaceEmbeddingModel(model_name="wont-load-scenario")
        model.model = None # Ensure model is None

        embedding = await model.generate_embedding("test text")
        assert embedding == []
        assert "Embedding model is not loaded (self.model is None)" in caplog.text

    @pytest.mark.asyncio
    async def test_hf_generate_embedding_success(self, mock_external_libs):
        """Tests successful embedding generation."""
        model = HuggingFaceEmbeddingModel(model_name="good-model")
        # mock_external_libs already set up model.model to be a MagicMock that returns [0.1,0.2,0.3]

        text_to_embed = "This is a test sentence."
        embedding = await model.generate_embedding(text_to_embed)
        assert embedding == [0.1, 0.2, 0.3]
        model.model.encode.assert_called_once_with(text_to_embed, convert_to_tensor=False)

    @pytest.mark.asyncio
    async def test_hf_generate_embedding_empty_or_invalid_text(self, caplog):
        model = HuggingFaceEmbeddingModel(model_name="good-model")
        assert await model.generate_embedding("") == []
        assert "Invalid input for embedding: Text is empty or not a string" in caplog.text
        caplog.clear()
        assert await model.generate_embedding(None) == [] # type: ignore
        assert "Invalid input for embedding: Text is empty or not a string" in caplog.text


# --- Test StubLTM (FAISS Integration) ---

@pytest.fixture
def mock_faiss_instance(monkeypatch):
    # Specific mock for faiss.index operations within StubLTM tests
    # This overrides the global mock_faiss_index_instance if more specific behavior is needed per test
    idx_instance = MagicMock()
    idx_instance.ntotal = 0
    idx_instance.search.return_value = (MagicMock(size=0), MagicMock(size=0)) # distances, indices
    idx_instance.add_with_ids.side_effect = lambda x, y: setattr(idx_instance, 'ntotal', idx_instance.ntotal + len(x))

    # Make faiss.IndexFlatL2 and faiss.IndexIDMap2 return this specific instance for StubLTM tests
    mock_faiss_module = contextkernel.core_logic.llm_retriever.faiss
    monkeypatch.setattr(mock_faiss_module, "IndexFlatL2", MagicMock(return_value=idx_instance))
    monkeypatch.setattr(mock_faiss_module, "IndexIDMap2", MagicMock(return_value=idx_instance))
    return idx_instance

@pytest.fixture
def mock_whoosh_for_ltm(monkeypatch):
    mock_ix = MagicMock()
    mock_writer_instance = MagicMock()
    mock_ix.writer.return_value = mock_writer_instance
    return mock_ix, mock_writer_instance


class TestStubLTM:
    def test_ltm_init_no_path(self, mock_faiss_instance):
        """Tests StubLTM initialization without a FAISS path."""
        ltm = StubLTM()
        assert ltm.index is None # Should be None as no path provided and no docs added
        assert ltm.faiss_index_path is None

    def test_ltm_init_with_path_not_exists(self, monkeypatch, mock_faiss_instance):
        """Tests StubLTM initialization with a FAISS path that doesn't exist."""
        monkeypatch.setattr("os.path.exists", MagicMock(return_value=False))
        ltm = StubLTM(faiss_index_path="non_existent_path.faiss")
        assert ltm.index is None
        mock_faiss_module = contextkernel.core_logic.llm_retriever.faiss
        mock_faiss_module.read_index.assert_not_called()

    def test_ltm_init_with_path_exists_loads_index(self, monkeypatch, mock_faiss_instance):
        """Tests StubLTM initialization with an existing FAISS path."""
        monkeypatch.setattr("os.path.exists", MagicMock(return_value=True))
        mock_read_index = MagicMock(return_value=mock_faiss_instance) # read_index returns our mock index

        mock_faiss_module = contextkernel.core_logic.llm_retriever.faiss
        monkeypatch.setattr(mock_faiss_module, "read_index", mock_read_index)

        ltm = StubLTM(faiss_index_path="existing_path.faiss")
        assert ltm.index == mock_faiss_instance
        mock_read_index.assert_called_once_with("existing_path.faiss")

    def test_ltm_init_faiss_not_available(self, monkeypatch, caplog):
        """Tests StubLTM initialization when FAISS library is not available."""
        monkeypatch.setattr("contextkernel.core_logic.llm_retriever.faiss", None)
        ltm = StubLTM()
        assert ltm.index is None
        assert "FAISS library not installed" in caplog.text

    @pytest.mark.asyncio
    async def test_ltm_add_document_creates_index_and_adds(self, mock_faiss_instance, default_config):
        """Tests adding a document creates FAISS index if none exists."""
        ltm = StubLTM(retriever_config=default_config) # No Whoosh for this test
        ltm.index = None # Explicitly start with no index

        doc_id = "doc1"
        text_content = "Test doc 1"
        item_metadata = {"source": "test", "doc_id": doc_id} # ensure doc_id is in metadata for storage
        embedding = [0.1, 0.2, 0.3]

        await ltm.add_document(doc_id=doc_id, text_content=text_content, embedding=embedding, metadata=item_metadata)

        assert ltm.index is not None
        # mock_faiss_instance is what ltm.index becomes due to fixture
        mock_faiss_instance.add_with_ids.assert_called_once()
        assert ltm.index.ntotal == 1
        assert ltm.doc_id_to_internal_idx["doc1"] == 0
        assert ltm.internal_idx_to_doc_item[0].content == "Test doc 1"

    @pytest.mark.asyncio
    async def test_ltm_add_document_to_existing_index(self, mock_faiss_instance, default_config):
        ltm = StubLTM(retriever_config=default_config)
        ltm.index = mock_faiss_instance

        await ltm.add_document(doc_id="id1", text_content="Doc 1", embedding=[0.1,0.1], metadata={"source":"test", "doc_id":"id1"})
        await ltm.add_document(doc_id="id2", text_content="Doc 2", embedding=[0.2,0.2], metadata={"source":"test", "doc_id":"id2"})

        assert mock_faiss_instance.add_with_ids.call_count == 2 # Each successful add calls this
        assert ltm.index.ntotal == 2
        assert ltm.doc_id_to_internal_idx["id2"] == 1

    @pytest.mark.asyncio
    async def test_ltm_add_document_with_whoosh_indexing(self, mock_faiss_instance, mock_whoosh_for_ltm, default_config):
        """Tests that add_document also calls Whoosh writer if configured."""
        mock_wh_ix, mock_wh_writer = mock_whoosh_for_ltm
        config_with_whoosh = default_config
        config_with_whoosh.keyword_search_enabled = True # Ensure Whoosh is enabled

        ltm = StubLTM(whoosh_ix=mock_wh_ix, retriever_config=config_with_whoosh)
        ltm.index = mock_faiss_instance

        doc_id_whoosh = "wh001"
        text_content_whoosh = "Whoosh test content"
        metadata_whoosh = {"source": "test", "doc_id": doc_id_whoosh}
        embedding_whoosh = [0.5, 0.5]
        await ltm.add_document(doc_id=doc_id_whoosh, text_content=text_content_whoosh, embedding=embedding_whoosh, metadata=metadata_whoosh)

        mock_wh_writer.add_document.assert_called_once_with(doc_id=doc_id_whoosh, content=text_content_whoosh)
        mock_wh_writer.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_ltm_search_empty_index(self, default_config):
        ltm = StubLTM(retriever_config=default_config)
        # ltm.index is None initially
        results = await ltm.search(query_embedding=[0.1,0.2], top_k=3)
        assert results == []

    @pytest.mark.asyncio
    async def test_ltm_search_faiss_success(self, mock_faiss_instance, default_config):
        ltm = StubLTM(retriever_config=default_config)
        ltm.index = mock_faiss_instance # Use the auto-mocked one

        # Populate internal_idx_to_doc_item as if docs were added
        item1 = RetrievedItem(content="Result doc1", source="ltm_test", metadata={"doc_id": "res1"})
        ltm.internal_idx_to_doc_item[0] = item1
        ltm.doc_id_to_internal_idx["res1"] = 0

        # Configure mock_faiss_instance.search to return specific values
        # Distances (L2), Indices. Indices should map to keys in internal_idx_to_doc_item
        mock_distances = contextkernel.core_logic.llm_retriever.np.array([[0.5]], dtype="float32")
        mock_indices = contextkernel.core_logic.llm_retriever.np.array([[0]], dtype="int64") # Index 0 matches item1
        mock_faiss_instance.search.return_value = (mock_distances, mock_indices)
        mock_faiss_instance.ntotal = 1 # Indicate one item in index

        results = await ltm.search(query_embedding=[0.1,0.1,0.1], top_k=1)

        assert len(results) == 1
        assert results[0].content == "Result doc1"
        assert results[0].metadata["doc_id"] == "res1"
        # Score for L2 distance 0.5: 1 / (1 + 0.5) = 0.666...
        assert abs(results[0].score - (1.0 / (1.0 + 0.5))) < 0.001

    @pytest.mark.asyncio
    async def test_ltm_save_index(self, monkeypatch, mock_faiss_instance):
        ltm = StubLTM(faiss_index_path="test_save.faiss")
        ltm.index = mock_faiss_instance # Assign a mock index to be saved

        mock_write_index = MagicMock()
        monkeypatch.setattr("contextkernel.core_logic.llm_retriever.faiss.write_index", mock_write_index)

        await ltm.save_index()
        mock_write_index.assert_called_once_with(mock_faiss_instance, "test_save.faiss")

    @pytest.mark.asyncio
    async def test_ltm_add_document_no_doc_id(self, caplog, default_config):
        ltm = StubLTM(retriever_config=default_config)
        item_no_id = RetrievedItem(content="No doc id here", source="test", metadata={})
        await ltm.add_document(item_no_id, [0.1,0.2])
        assert "Document item is missing 'doc_id' in metadata" in caplog.text
        assert ltm.index is None # Index should not be created if doc has no id for mapping.

# --- Test StubGraphDB (NetworkX Integration) ---

@pytest.fixture
def mock_nx_graph_fixture(monkeypatch):
    # Specific mock for nx.graph operations within StubGraphDB tests
    graph_instance = MagicMock()
    graph_instance.nodes = {}
    graph_instance.edges = []
    graph_instance.has_node.side_effect = lambda n: n in graph_instance.nodes
    graph_instance.add_node.side_effect = lambda n, **attrs: graph_instance.nodes.update({n: attrs})
    # A simplified add_edge mock. Real NetworkX handles multiple edges, etc.
    graph_instance.add_edge.side_effect = lambda u, v, **attrs: graph_instance.edges.append((u,v,attrs))

    # Make nx.Graph() return this specific instance for StubGraphDB tests
    mock_nx_module = contextkernel.core_logic.llm_retriever.nx
    monkeypatch.setattr(mock_nx_module, "Graph", MagicMock(return_value=graph_instance))
    return graph_instance

class TestStubGraphDB:
    def test_graphdb_init_no_path(self, mock_nx_graph_fixture):
        """Tests StubGraphDB initialization without a graph path."""
        db = StubGraphDB()
        assert db.graph is not None # Should be an empty graph by default from mock_nx_graph_fixture
        assert db.networkx_graph_path is None
        # mock_nx_graph_fixture ensures nx.Graph() was called.

    def test_graphdb_init_with_path_not_exists(self, monkeypatch, mock_nx_graph_fixture):
        """Tests StubGraphDB initialization with a graph path that doesn't exist."""
        monkeypatch.setattr("os.path.exists", MagicMock(return_value=False))
        db = StubGraphDB(networkx_graph_path="non_existent.gml")
        assert db.graph is not None # Still should be an empty graph
        mock_nx_module = contextkernel.core_logic.llm_retriever.nx
        mock_nx_module.read_gml.assert_not_called()
        mock_nx_module.read_graphml.assert_not_called()

    def test_graphdb_init_with_gml_path_exists(self, monkeypatch, mock_nx_graph_fixture):
        """Tests StubGraphDB initialization with an existing GML graph path."""
        monkeypatch.setattr("os.path.exists", MagicMock(return_value=True))
        mock_read_gml = MagicMock(return_value=mock_nx_graph_fixture)
        mock_nx_module = contextkernel.core_logic.llm_retriever.nx
        monkeypatch.setattr(mock_nx_module, "read_gml", mock_read_gml)

        db = StubGraphDB(networkx_graph_path="existing.gml")
        assert db.graph == mock_nx_graph_fixture
        mock_read_gml.assert_called_once_with("existing.gml")

    def test_graphdb_init_networkx_not_available(self, monkeypatch, caplog):
        """Tests StubGraphDB initialization when NetworkX library is not available."""
        monkeypatch.setattr("contextkernel.core_logic.llm_retriever.nx", None)
        db = StubGraphDB()
        assert db.graph is None
        assert "NetworkX library not installed" in caplog.text

    @pytest.mark.asyncio
    async def test_graphdb_add_node_and_relation(self, mock_nx_graph_fixture):
        """Tests adding nodes and relations to the graph."""
        db = StubGraphDB()
        db.graph = mock_nx_graph_fixture # Ensure it uses the test-specific graph mock

        await db.add_node("node1", name="Node One", type="person")
        await db.add_node("node2", name="Node Two", type="place")
        await db.add_relation("node1", "node2", type="VISITED", year=2023)

        mock_nx_graph_fixture.add_node.assert_any_call("node1", name="Node One", type="person")
        mock_nx_graph_fixture.add_node.assert_any_call("node2", name="Node Two", type="place")
        mock_nx_graph_fixture.add_edge.assert_called_once_with("node1", "node2", type="VISITED", year=2023)

    @pytest.mark.asyncio
    async def test_graphdb_search_node_by_id(self, mock_nx_graph_fixture):
        db = StubGraphDB()
        db.graph = mock_nx_graph_fixture

        # Setup mock graph state for search
        mock_nx_graph_fixture.has_node.return_value = True
        mock_nx_graph_fixture.nodes = {"node1": {"name": "Test Node", "attr": "val"}}
        mock_nx_graph_fixture.edges = MagicMock(return_value=[("node1", "other_node", {"type": "CONNECTS_TO"})])

        results = await db.search(query="node1", top_k=1)
        assert len(results) == 1
        assert results[0].source == "graph_db_nx_node_id_match"
        assert results[0].content["node_id"] == "node1"
        assert results[0].content["properties"]["name"] == "Test Node"

    @pytest.mark.asyncio
    async def test_graphdb_search_node_by_property(self, mock_nx_graph_fixture):
        db = StubGraphDB()
        db.graph = mock_nx_graph_fixture

        # Setup mock graph state
        mock_nx_graph_fixture.nodes = {
            "nodeA": {"name": "Alice", "occupation": "engineer"},
            "nodeB": {"name": "Bob", "occupation": "artist"}
        }
        # Ensure nodes(data=True) returns what's in mock_nx_graph_fixture.nodes
        mock_nx_graph_fixture.nodes.items.return_value = mock_nx_graph_fixture.nodes.items()
        # More accurate mocking for nodes(data=True)
        def mock_nodes_data_true(data=False):
            if data is True:
                return list(mock_nx_graph_fixture.nodes.items())
            return list(mock_nx_graph_fixture.nodes.keys())
        mock_nx_graph_fixture.nodes = MagicMock(side_effect=mock_nodes_data_true)
        mock_nx_graph_fixture.nodes.items = MagicMock(return_value=db.graph.nodes.items()) # Re-point after side_effect

        # Re-assign the .nodes attribute to a MagicMock that can handle being called with `(data=True)`
        # This is tricky because `graph.nodes` is both a property and can be called.
        # The global mock_nx_graph_instance.nodes might be too simple.
        # Let's refine the mock_nx_graph_fixture:

        # For this test, we'll directly patch the graph instance's .nodes attribute
        # to control what `self.graph.nodes(data=True)` returns.
        db.graph.nodes = MagicMock(return_value=[
            ("nodeA", {"name": "Alice", "occupation": "engineer"}),
            ("nodeB", {"name": "Bob", "occupation": "artist"})
        ])

        results = await db.search(query="name:Alice", top_k=1)
        assert len(results) == 1
        assert results[0].source == "graph_db_nx_property_match"
        assert results[0].content["properties"]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_graphdb_save_graph_gml(self, monkeypatch, mock_nx_graph_fixture):
        """Tests saving the graph in GML format."""
        db = StubGraphDB(networkx_graph_path="test_graph.gml")
        db.graph = mock_nx_graph_fixture # Assign the mocked graph

        mock_write_gml = MagicMock()
        mock_nx_module = contextkernel.core_logic.llm_retriever.nx
        monkeypatch.setattr(mock_nx_module, "write_gml", mock_write_gml)

        await db.save_graph()
        mock_write_gml.assert_called_once_with(mock_nx_graph_fixture, "test_graph.gml")

# --- Test LLMRetriever ---

@pytest.fixture
def mock_ltm_interface():
    ltm = MagicMock(spec=StubLTM) # Use spec to ensure it mocks StubLTM's interface
    ltm.search = AsyncMock(return_value=[]) # Default to no results
    # Mock attributes that LLMRetriever.__init__ tries to set
    ltm.whoosh_ix = None
    ltm.retriever_config = None
    return ltm

@pytest.fixture
def mock_graphdb_interface():
    graphdb = MagicMock(spec=StubGraphDB)
    graphdb.search = AsyncMock(return_value=[])
    return graphdb

@pytest.fixture
def llm_retriever(default_config, mock_ltm_interface, mock_graphdb_interface):
    """Fixture to create an LLMRetriever instance with mocked dependencies."""
    # Embedding model (HuggingFaceEmbeddingModel) is already mocked by mock_external_libs
    # CrossEncoder is also mocked by mock_external_libs
    # Whoosh index creation/opening is also mocked

    # Ensure that the mock_ltm_interface has whoosh_ix and retriever_config attributes
    # if they are accessed during LLMRetriever initialization.
    # The spec arg to MagicMock doesn't create these automatically if they are just attributes.
    # We can add them here if needed, or ensure __init__ checks with hasattr.
    # Based on current LLMRetriever.__init__, it uses hasattr, so this should be fine.

    retriever = LLMRetriever(
        retriever_config=default_config,
        ltm_interface=mock_ltm_interface,
        stm_interface=MagicMock(), # Basic mock for STM, not heavily used yet
        graphdb_interface=mock_graphdb_interface,
        query_llm=None # No query LLM for these tests yet
    )
    return retriever

class TestLLMRetriever:
    def test_retriever_init_components(self, llm_retriever, default_config, mock_ltm_interface):
        """Tests that components are initialized in LLMRetriever."""
        assert llm_retriever.retriever_config == default_config
        assert llm_retriever.ltm == mock_ltm_interface
        assert llm_retriever.embedding_model is not None # Mocked by global fixture

        if default_config.keyword_search_enabled:
            assert llm_retriever.whoosh_ix is not None # Mocked by global fixture
            # Check if it was passed to LTM
            assert mock_ltm_interface.whoosh_ix is not None
            assert mock_ltm_interface.retriever_config is not None
        else:
            assert llm_retriever.whoosh_ix is None

        if default_config.cross_encoder_model_name: # e.g. "cross-encoder-model"
            assert llm_retriever.cross_encoder is not None
            # Check that CrossEncoder constructor was called
            mock_external_libs_cross_encoder = contextkernel.core_logic.llm_retriever.CrossEncoder
            mock_external_libs_cross_encoder.assert_called_with(default_config.cross_encoder_model_name)
        else:
            assert llm_retriever.cross_encoder is None


    @pytest.mark.asyncio
    async def test_retriever_preprocess_and_embed_query(self, llm_retriever):
        """Tests query embedding process."""
        query = "Test query for embedding"
        # llm_retriever.embedding_model is a MagicMock from HuggingFaceEmbeddingModel
        # which itself uses the mocked SentenceTransformer that returns [0.1,0.2,0.3]
        llm_retriever.embedding_model.generate_embedding = AsyncMock(return_value=[0.1,0.2,0.3])

        embedding = await llm_retriever._preprocess_and_embed_query(query)
        assert embedding == [0.1, 0.2, 0.3]
        llm_retriever.embedding_model.generate_embedding.assert_called_once_with(query)

    @pytest.mark.asyncio
    async def test_retriever_search_vector_store(self, llm_retriever, mock_ltm_interface):
        """Tests calling LTM search."""
        query_embedding = [0.1, 0.2, 0.3]
        mock_ltm_interface.search.return_value = [
            RetrievedItem(content="LTM Result 1", source="ltm", score=0.9)
        ]
        results = await llm_retriever._search_vector_store(query_embedding, top_k=1)
        assert len(results) == 1
        assert results[0].content == "LTM Result 1"
        mock_ltm_interface.search.assert_called_once_with(query_embedding=query_embedding, top_k=1, filters=None)

    @pytest.mark.asyncio
    async def test_retriever_search_keyword_enabled(self, llm_retriever, monkeypatch):
        """Tests keyword search when enabled."""
        llm_retriever.retriever_config.keyword_search_enabled = True
        # Ensure whoosh_ix is the mocked one from global fixture
        mock_searcher_instance = MagicMock()
        mock_searcher_instance.search.return_value = [ # Simulate Whoosh Hit objects
            {"doc_id": "whoosh_doc1", "content": "Keyword content", "score": 0.85}
        ]
        # The context manager `with self.whoosh_ix.searcher() as searcher:` needs specific mocking
        mock_whoosh_index_on_retriever = llm_retriever.whoosh_ix # This is the globally mocked Whoosh index
        mock_searcher_cm = MagicMock() # Mock for the context manager object
        mock_searcher_cm.__enter__.return_value = mock_searcher_instance # __enter__ returns the searcher
        mock_whoosh_index_on_retriever.searcher.return_value = mock_searcher_cm

        # Mock QueryParser
        mock_qparser_instance = MagicMock()
        mock_qparser_instance.parse.return_value = "parsed_query_obj"
        mock_qparser_constructor = MagicMock(return_value=mock_qparser_instance)
        monkeypatch.setattr("contextkernel.core_logic.llm_retriever.QueryParser", mock_qparser_constructor)

        results = await llm_retriever._search_keyword("test query", top_k=1)

        assert len(results) == 1
        assert results[0].content == "Keyword content"
        assert results[0].source == "keyword_search"
        mock_qparser_constructor.assert_called_once_with("content", schema=llm_retriever.whoosh_ix.schema)
        mock_qparser_instance.parse.assert_called_once_with("test query")
        mock_searcher_instance.search.assert_called_once_with("parsed_query_obj", limit=1)


    @pytest.mark.asyncio
    async def test_retriever_search_keyword_disabled(self, llm_retriever):
        """Tests that keyword search returns empty if disabled."""
        llm_retriever.retriever_config.keyword_search_enabled = False
        results = await llm_retriever._search_keyword("test", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_retriever_consolidate_simple_aggregation(self, llm_retriever):
        """Tests simple aggregation and sorting of results."""
        results_collection = [
            [RetrievedItem(content="Res1", source="ltm", score=0.8, metadata={"doc_id":"1"})],
            [RetrievedItem(content="Res2", source="graph", score=0.9, metadata={"doc_id":"2"})],
            [RetrievedItem(content="Res1 Duplicate", source="keyword", score=0.85, metadata={"doc_id":"1"})], # Duplicate
        ]
        # Pass dummy query as it's now required by _consolidate_and_rank_results
        consolidated = await llm_retriever._consolidate_and_rank_results("dummy query", results_collection)
        assert len(consolidated) == 2 # Res1 (keyword version due to higher score), Res2
        assert consolidated[0].content == "Res2" # Highest score
        assert consolidated[1].content == "Res1 Duplicate" # Higher score duplicate chosen

    @pytest.mark.asyncio
    async def test_retriever_consolidate_with_cross_encoder(self, llm_retriever, default_config, monkeypatch):
        """Tests cross-encoder re-ranking logic."""
        # Enable cross-encoder for this test
        default_config.cross_encoder_model_name = "test-cross-encoder"
        # Re-init retriever with this config to load the cross-encoder mock
        # (The llm_retriever fixture uses default_config as it is at the start of the test run)
        # Instead of re-init, we can directly set the cross_encoder on the existing instance
        mock_cross_encoder_instance = MagicMock()
        # Let new scores reverse the original order for a clear test
        mock_cross_encoder_instance.predict.return_value = [0.7, 0.95] # Score for Item1, Item2

        mock_ce_constructor = MagicMock(return_value=mock_cross_encoder_instance)
        monkeypatch.setattr("contextkernel.core_logic.llm_retriever.CrossEncoder", mock_ce_constructor)

        # Manually create a new retriever or update the existing one's cross_encoder
        retriever_for_ce_test = LLMRetriever(default_config, MagicMock(), MagicMock(), MagicMock())
        assert retriever_for_ce_test.cross_encoder is not None # Should be the mocked one

        results_collection = [[
            RetrievedItem(content="Item1", source="ltm", score=0.9, metadata={"doc_id":"1"}), # Originally higher
            RetrievedItem(content="Item2", source="ltm", score=0.8, metadata={"doc_id":"2"})  # Originally lower
        ]]

        consolidated = await retriever_for_ce_test._consolidate_and_rank_results(
            query="test query for ce",
            results_collection=results_collection,
            top_k_for_cross_encoder=2 # Rerank both
        )
        assert len(consolidated) == 2
        assert consolidated[0].content == "Item2" # Item2 should now be first due to higher CE score (0.95)
        assert consolidated[1].content == "Item1" # Item1 has CE score 0.7
        assert consolidated[0].score == 0.95
        assert consolidated[1].score == 0.7
        mock_cross_encoder_instance.predict.assert_called_once()
        # Check pairs sent to cross-encoder: (query, content)
        expected_pairs = [["test query for ce", "Item1"], ["test query for ce", "Item2"]]
        actual_pairs = mock_cross_encoder_instance.predict.call_args[0][0]
        assert actual_pairs == expected_pairs


    @pytest.mark.asyncio
    async def test_retriever_retrieve_all_strategy(self, llm_retriever, mock_ltm_interface, mock_graphdb_interface, monkeypatch):
        """Tests the 'all' retrieval strategy."""
        query = "Retrieve everything"

        # Mock responses from each source
        llm_retriever.embedding_model.generate_embedding = AsyncMock(return_value=[0.5]*5)
        mock_ltm_interface.search.return_value = [RetrievedItem(content="LTM data", source="ltm", score=0.7, metadata={"doc_id":"ltm1"})]
        mock_graphdb_interface.search.return_value = [RetrievedItem(content="Graph data", source="graph", score=0.8, metadata={"doc_id":"graph1"})]

        # Mock keyword search part for 'all' strategy
        mock_searcher_instance = MagicMock()
        mock_searcher_instance.search.return_value = [{"doc_id": "kw1", "content": "Keyword data", "score": 0.9}]
        mock_searcher_cm = MagicMock(); mock_searcher_cm.__enter__.return_value = mock_searcher_instance
        llm_retriever.whoosh_ix.searcher.return_value = mock_searcher_cm
        monkeypatch.setattr("contextkernel.core_logic.llm_retriever.QueryParser", MagicMock(return_value=MagicMock(parse=MagicMock(return_value="p"))))

        response = await llm_retriever.retrieve(query, retrieval_strategy="all")

        assert len(response.items) == 3
        assert response.items[0].content == "Keyword data" # Highest score
        mock_ltm_interface.search.assert_called_once()
        mock_graphdb_interface.search.assert_called_once()
        llm_retriever.whoosh_ix.searcher.assert_called_once() # Check if Whoosh searcher was used


    @pytest.mark.asyncio
    async def test_retriever_retrieve_embedding_failure(self, llm_retriever, caplog):
        """Tests retrieval when embedding generation fails."""
        llm_retriever.embedding_model.generate_embedding = AsyncMock(return_value=[]) # Simulate failure

        response = await llm_retriever.retrieve("query", retrieval_strategy="vector_only")
        assert len(response.items) == 0
        assert "Failed to generate query embedding" in response.message
        assert "Failed to generate query embedding" in caplog.text
