# Prototype: In-memory system for transient data. Functionally complete for its defined scope.
import time
import uuid

class WorkingMemorySystem:
    """
    A system for managing temporary, in-memory 'notes' or pieces of information,
    simulating a working memory for an agent or application.
    """

    def __init__(self, default_decay_rate_seconds: float = None):
        """
        Initializes the Working Memory System.

        Args:
            default_decay_rate_seconds (float, optional): A parameter for future use in
                                                        relevance decay calculations. Not actively used
                                                        in the current basic implementation.
        """
        self.memory_store = {}  # In-memory dictionary to store notes
        self.default_decay_rate_seconds = default_decay_rate_seconds # Not used yet

    def add_or_update_note(self, content: any, origin: str, note_id: str = None, relevance_score: float = 1.0) -> str:
        """
        Adds a new note or updates an existing one in the memory store.

        Args:
            content (any): The actual data or thought to be stored.
            origin (str): Identifier for the source of this note (e.g., 'user_input', 'module_A').
            note_id (str, optional): The ID of the note to update. If None or not found, a new note is created.
            relevance_score (float, optional): The initial or updated relevance score of the note. Defaults to 1.0.

        Returns:
            str: The ID of the added or updated note.
        """
        current_time = time.time()

        if note_id and note_id in self.memory_store:
            # Update existing note
            note = self.memory_store[note_id]
            note['content'] = content
            note['timestamp'] = current_time  # Update timestamp to reflect modification
            note['relevance_score'] = relevance_score
            note['origin'] = origin # Origin might also be updated
            note['last_accessed_at'] = current_time
        else:
            # Create new note
            new_id = uuid.uuid4().hex
            note = {
                'id': new_id,
                'content': content,
                'timestamp': current_time,
                'relevance_score': relevance_score,
                'origin': origin,
                'last_accessed_at': current_time,
                # 'decay_score': 1.0 # Example for future use
            }
            self.memory_store[new_id] = note
            note_id = new_id
        
        return note_id

    def get_note_by_id(self, note_id: str) -> dict | None:
        """
        Retrieves a note by its ID and updates its last access time.

        Args:
            note_id (str): The ID of the note to retrieve.

        Returns:
            dict | None: The note dictionary if found, otherwise None.
        """
        if note_id in self.memory_store:
            note = self.memory_store[note_id]
            note['last_accessed_at'] = time.time()
            return note
        return None

    def get_recent_notes(self, count: int) -> list[dict]:
        """
        Retrieves the most recently added or updated notes.

        Args:
            count (int): The maximum number of recent notes to retrieve.

        Returns:
            list[dict]: A list of note dictionaries, sorted by timestamp in descending order.
        """
        if count <= 0:
            return []
        # Sort notes by timestamp (creation/update time) in descending order
        sorted_notes = sorted(self.memory_store.values(), key=lambda x: x['timestamp'], reverse=True)
        return sorted_notes[:count]

    def get_relevant_notes(self, min_relevance: float) -> list[dict]:
        """
        Retrieves notes with a relevance score greater than or equal to the specified minimum.
        Updates 'last_accessed_at' for all returned notes.

        Args:
            min_relevance (float): The minimum relevance score for notes to be included.

        Returns:
            list[dict]: A list of note dictionaries meeting the relevance criteria,
                        sorted by relevance_score in descending order.
        """
        current_time = time.time()
        relevant_notes = []
        for note in self.memory_store.values():
            if note['relevance_score'] >= min_relevance:
                note['last_accessed_at'] = current_time
                relevant_notes.append(note)
        
        # Sort by relevance score in descending order
        relevant_notes.sort(key=lambda x: x['relevance_score'], reverse=True)
        return relevant_notes

    def decay_relevance(self, decay_factor: float = 0.05, time_threshold_seconds: float = 3600):
        """
        Reduces the relevance_score of notes based on time since last access or creation.
        This is a basic implementation placeholder. More sophisticated models could be used.

        Args:
            decay_factor (float): The factor by which relevance is reduced (e.g., 0.05 for 5% reduction).
                                  Should be between 0 and 1.
            time_threshold_seconds (float): Only decay notes not accessed within this duration.
        """
        if not (0 <= decay_factor <= 1):
            print("Decay factor must be between 0 and 1.")
            return

        current_time = time.time()
        notes_to_decay = []
        for note_id, note in self.memory_store.items():
            # Decay if not accessed recently
            if current_time - note['last_accessed_at'] > time_threshold_seconds:
                notes_to_decay.append(note_id)
        
        for note_id in notes_to_decay:
            new_relevance = self.memory_store[note_id]['relevance_score'] * (1 - decay_factor)
            self.memory_store[note_id]['relevance_score'] = max(0, new_relevance) # Ensure relevance doesn't go below 0
            # print(f"Decayed relevance of note {note_id} to {self.memory_store[note_id]['relevance_score']}")


    def remove_note(self, note_id: str) -> bool:
        """
        Removes a note from the memory store.

        Args:
            note_id (str): The ID of the note to remove.

        Returns:
            bool: True if the note was found and removed, False otherwise.
        """
        if note_id in self.memory_store:
            del self.memory_store[note_id]
            return True
        return False

    def get_all_notes(self) -> list[dict]:
        """
        Retrieves all notes currently in the memory store.
        Mainly for debugging or inspection.

        Returns:
            list[dict]: A list of all note dictionaries.
        """
        return list(self.memory_store.values())


if __name__ == '__main__':
    print("Initializing Working Memory System...")
    wms = WorkingMemorySystem(default_decay_rate_seconds=60*60) # Example: 1 hour decay rate (not actively used by decay_relevance yet)

    print("\n--- Adding Notes ---")
    note1_id = wms.add_or_update_note(content="Initial idea about project X: Core components definition.", origin="brainstorm_module", relevance_score=0.8)
    print(f"Added note 1 (ID: {note1_id})")
    time.sleep(0.1) # Ensure distinct timestamps for sorting
    note2_id = wms.add_or_update_note(content="User query: 'What are the key features of X?'", origin="input_handler", relevance_score=0.95)
    print(f"Added note 2 (ID: {note2_id})")
    time.sleep(0.1)
    note3_id = wms.add_or_update_note(content="System status: All services nominal.", origin="monitoring_service", relevance_score=0.5)
    print(f"Added note 3 (ID: {note3_id})")

    print("\n--- Getting Note by ID ---")
    retrieved_note1 = wms.get_note_by_id(note1_id)
    print(f"Retrieved note 1: {retrieved_note1['content'] if retrieved_note1 else 'Not found'}")
    print(f"Note 1 last_accessed_at: {retrieved_note1['last_accessed_at'] if retrieved_note1 else 'N/A'}")

    print("\n--- Updating Note ---")
    wms.add_or_update_note(content="Initial idea about project X: Core components defined and user interface mockups started.", origin="brainstorm_module_update", note_id=note1_id, relevance_score=0.85)
    updated_note1 = wms.get_note_by_id(note1_id)
    print(f"Updated note 1 content: {updated_note1['content']}")
    print(f"Updated note 1 relevance: {updated_note1['relevance_score']}")
    print(f"Updated note 1 timestamp: {updated_note1['timestamp']}")


    print("\n--- Getting Recent Notes (Top 2) ---")
    recent_notes = wms.get_recent_notes(count=2)
    print(f"Found {len(recent_notes)} recent notes:")
    for note in recent_notes:
        print(f"  ID: {note['id']}, Content: '{note['content']}', Timestamp: {note['timestamp']}")

    print("\n--- Getting Relevant Notes (min_relevance=0.8) ---")
    relevant_notes = wms.get_relevant_notes(min_relevance=0.8)
    print(f"Found {len(relevant_notes)} relevant notes (>=0.8):")
    for note in relevant_notes:
        print(f"  ID: {note['id']}, Content: '{note['content']}', Relevance: {note['relevance_score']}, Last Accessed: {note['last_accessed_at']}")

    print("\n--- Decaying Relevance (example) ---")
    print("Simulating time passage for decay...")
    # Manually adjust last_accessed_at for testing decay on one note
    if note3_id in wms.memory_store:
        wms.memory_store[note3_id]['last_accessed_at'] = time.time() - 7200 # Simulate 2 hours since last access
    print(f"Note 3 relevance before decay: {wms.memory_store.get(note3_id, {}).get('relevance_score')}")
    wms.decay_relevance(decay_factor=0.1, time_threshold_seconds=3600) # 10% decay if not accessed in 1hr
    print(f"Note 3 relevance after decay: {wms.memory_store.get(note3_id, {}).get('relevance_score')}")
    print(f"Note 1 relevance (should not decay): {wms.memory_store.get(note1_id, {}).get('relevance_score')}")


    print("\n--- Removing Note ---")
    removed = wms.remove_note(note2_id)
    print(f"Note 2 (ID: {note2_id}) removed: {removed}")
    print(f"Trying to retrieve removed note 2: {wms.get_note_by_id(note2_id)}")

    print("\n--- Final state of all notes ---")
    all_notes = wms.get_all_notes()
    if all_notes:
        for note in all_notes:
            print(f"  ID: {note['id']}, Content: '{note['content']}', Relevance: {note['relevance_score']}")
    else:
        print("No notes in memory.")

    print("\nWorking Memory System demo complete.")
