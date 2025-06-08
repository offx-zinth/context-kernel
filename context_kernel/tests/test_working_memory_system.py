import unittest
import time
import uuid
from unittest.mock import patch

# Adjust import path as necessary
from context_kernel.working_memory_system import WorkingMemorySystem

class TestWorkingMemorySystem(unittest.TestCase):

    def setUp(self):
        """Set up for each test."""
        self.wms = WorkingMemorySystem(default_decay_rate_seconds=3600)

    def test_init(self):
        """Test basic initialization."""
        self.assertIsNotNone(self.wms.memory_store)
        self.assertEqual(self.wms.default_decay_rate_seconds, 3600)

    def test_add_new_note(self):
        """Test adding a new note."""
        content = "This is a new note."
        origin = "test_origin_new"
        relevance = 0.85

        # Mock time.time() to control timestamps
        current_time_mock = time.time()
        with patch('time.time', return_value=current_time_mock):
            note_id = self.wms.add_or_update_note(content=content, origin=origin, relevance_score=relevance)

        self.assertIsNotNone(note_id)
        self.assertIn(note_id, self.wms.memory_store)
        
        note = self.wms.memory_store[note_id]
        self.assertEqual(note['id'], note_id)
        self.assertEqual(note['content'], content)
        self.assertEqual(note['origin'], origin)
        self.assertEqual(note['relevance_score'], relevance)
        self.assertEqual(note['timestamp'], current_time_mock)
        self.assertEqual(note['last_accessed_at'], current_time_mock)

    def test_update_existing_note(self):
        """Test updating an existing note."""
        content_initial = "Initial content."
        origin_initial = "test_origin_initial"
        
        initial_time_mock = time.time() - 100 # ensure timestamp is older
        with patch('time.time', return_value=initial_time_mock):
            note_id = self.wms.add_or_update_note(content=content_initial, origin=origin_initial)

        content_updated = "Updated content."
        origin_updated = "test_origin_updated"
        relevance_updated = 0.99
        
        update_time_mock = time.time() # Newer timestamp
        with patch('time.time', return_value=update_time_mock):
            updated_note_id = self.wms.add_or_update_note(
                note_id=note_id,
                content=content_updated,
                origin=origin_updated,
                relevance_score=relevance_updated
            )

        self.assertEqual(note_id, updated_note_id)
        self.assertIn(note_id, self.wms.memory_store)
        
        note = self.wms.memory_store[note_id]
        self.assertEqual(note['content'], content_updated)
        self.assertEqual(note['origin'], origin_updated)
        self.assertEqual(note['relevance_score'], relevance_updated)
        self.assertEqual(note['timestamp'], update_time_mock, "Timestamp should update on modification.")
        self.assertEqual(note['last_accessed_at'], update_time_mock)

    def test_get_note_by_id(self):
        """Test retrieving an existing note and updating its last_accessed_at."""
        content = "Note to be retrieved."
        origin = "test_get_id"
        
        creation_time = time.time() - 50
        with patch('time.time', return_value=creation_time):
            note_id = self.wms.add_or_update_note(content=content, origin=origin)
        
        self.assertEqual(self.wms.memory_store[note_id]['last_accessed_at'], creation_time)

        access_time = time.time() # Current time for access
        with patch('time.time', return_value=access_time):
            retrieved_note = self.wms.get_note_by_id(note_id)

        self.assertIsNotNone(retrieved_note)
        self.assertEqual(retrieved_note['id'], note_id)
        self.assertEqual(retrieved_note['content'], content)
        self.assertEqual(retrieved_note['last_accessed_at'], access_time, "last_accessed_at should update.")

    def test_get_note_by_id_non_existent(self):
        """Test attempting to retrieve a non-existent note."""
        non_existent_id = uuid.uuid4().hex
        retrieved_note = self.wms.get_note_by_id(non_existent_id)
        self.assertIsNone(retrieved_note)

    def test_get_recent_notes(self):
        """Test retrieving recent notes, checking sorting and count."""
        # Add a few notes with varying timestamps
        with patch('time.time') as mock_time:
            mock_time.return_value = time.time() - 200
            self.wms.add_or_update_note("Old note", "test_recent")
            
            mock_time.return_value = time.time() - 50
            note2_id = self.wms.add_or_update_note("Newer note", "test_recent")
            
            mock_time.return_value = time.time() - 100
            self.wms.add_or_update_note("Middle note", "test_recent")
            
            mock_time.return_value = time.time() - 10 # Most recent
            note4_id = self.wms.add_or_update_note("Most recent note", "test_recent")

        recent_2 = self.wms.get_recent_notes(count=2)
        self.assertEqual(len(recent_2), 2)
        self.assertEqual(recent_2[0]['id'], note4_id) # Most recent first
        self.assertEqual(recent_2[1]['id'], note2_id)

        recent_all = self.wms.get_recent_notes(count=10) # Get all (more than exist)
        self.assertEqual(len(recent_all), 4)
        self.assertEqual(recent_all[0]['id'], note4_id)

        recent_none = self.wms.get_recent_notes(count=0)
        self.assertEqual(len(recent_none),0)


    def test_get_relevant_notes(self):
        """Test filtering by relevance, sorting, and last_accessed_at updates."""
        access_time_mock = time.time()

        with patch('time.time', return_value=time.time() - 100): # Set initial creation/access time
            self.wms.add_or_update_note("Low relevance", "test_relevant", relevance_score=0.3)
            note_high_id = self.wms.add_or_update_note("High relevance", "test_relevant", relevance_score=0.9)
            self.wms.add_or_update_note("Medium relevance", "test_relevant", relevance_score=0.6)
            note_another_high_id = self.wms.add_or_update_note("Another high relevance", "test_relevant", relevance_score=0.9) # Same as note_high

        # Ensure last_accessed_at is old before calling get_relevant_notes
        self.assertLess(self.wms.memory_store[note_high_id]['last_accessed_at'], access_time_mock)

        with patch('time.time', return_value=access_time_mock): # Mock time for the access update
            relevant_notes = self.wms.get_relevant_notes(min_relevance=0.5)

        self.assertEqual(len(relevant_notes), 3)
        # Notes should be sorted by relevance_score descending
        self.assertTrue(relevant_notes[0]['relevance_score'] >= relevant_notes[1]['relevance_score'])
        self.assertTrue(relevant_notes[1]['relevance_score'] >= relevant_notes[2]['relevance_score'])
        self.assertEqual(relevant_notes[0]['relevance_score'], 0.9)
        self.assertEqual(relevant_notes[1]['relevance_score'], 0.9) # Could be note_high or note_another_high
        self.assertEqual(relevant_notes[2]['relevance_score'], 0.6)
        
        # Check that last_accessed_at was updated for returned notes
        for note in relevant_notes:
            self.assertEqual(self.wms.memory_store[note['id']]['last_accessed_at'], access_time_mock)
        
        # Check that the low relevance note was not affected (its last_accessed_at should be old)
        low_rel_note_id = [nid for nid, n in self.wms.memory_store.items() if n['content'] == "Low relevance"][0]
        self.assertLess(self.wms.memory_store[low_rel_note_id]['last_accessed_at'], access_time_mock)


    def test_remove_note(self):
        """Test removing a note."""
        note_id = self.wms.add_or_update_note("Note to be removed", "test_remove")
        self.assertIn(note_id, self.wms.memory_store)

        removed = self.wms.remove_note(note_id)
        self.assertTrue(removed)
        self.assertNotIn(note_id, self.wms.memory_store)

        removed_again = self.wms.remove_note(note_id) # Try removing non-existent
        self.assertFalse(removed_again)
        
        non_existent_id = uuid.uuid4().hex
        removed_non_existent = self.wms.remove_note(non_existent_id)
        self.assertFalse(removed_non_existent)

    def test_decay_relevance(self):
        """Test the basic relevance decay logic."""
        # Add a note that should decay and one that shouldn't
        decay_time_threshold = 3600  # As per default in WorkingMemorySystem method
        decay_factor = 0.1

        current_time = time.time()
        
        with patch('time.time', return_value=current_time - (decay_time_threshold + 100)): # Accessed long ago
            note_to_decay_id = self.wms.add_or_update_note("Decay me", "test_decay", relevance_score=1.0)
        
        with patch('time.time', return_value=current_time - 100): # Accessed recently
            note_not_to_decay_id = self.wms.add_or_update_note("Don't decay me", "test_decay", relevance_score=1.0)

        # Manually set last_accessed_at to ensure they are distinct for the test condition
        self.wms.memory_store[note_to_decay_id]['last_accessed_at'] = current_time - (decay_time_threshold + 100)
        self.wms.memory_store[note_not_to_decay_id]['last_accessed_at'] = current_time - 100


        # Call decay_relevance, all notes' last_accessed_at are now effectively 'current_time' for comparison
        with patch('time.time', return_value=current_time):
            self.wms.decay_relevance(decay_factor=decay_factor, time_threshold_seconds=decay_time_threshold)

        self.assertLess(self.wms.memory_store[note_to_decay_id]['relevance_score'], 1.0)
        self.assertEqual(self.wms.memory_store[note_to_decay_id]['relevance_score'], 1.0 * (1 - decay_factor))
        
        self.assertEqual(self.wms.memory_store[note_not_to_decay_id]['relevance_score'], 1.0) # Should not have decayed

    def test_decay_relevance_invalid_factor(self):
        """Test decay_relevance with an invalid decay factor."""
        note_id = self.wms.add_or_update_note("Test note", "decay_factor_test", relevance_score=1.0)
        initial_relevance = self.wms.memory_store[note_id]['relevance_score']

        with patch('builtins.print') as mock_print:
            self.wms.decay_relevance(decay_factor=1.5) # Invalid factor > 1
            self.wms.decay_relevance(decay_factor=-0.5) # Invalid factor < 0
        
        # Relevance should not change
        self.assertEqual(self.wms.memory_store[note_id]['relevance_score'], initial_relevance)
        mock_print.assert_any_call("Decay factor must be between 0 and 1.")


if __name__ == '__main__':
    unittest.main()
