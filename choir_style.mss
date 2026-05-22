<?xml version="1.0" encoding="UTF-8"?>
<museScore version="4.40">
  <Style>
    <!-- Eliminate horizontal offset between voices on the same staff.
         This makes voice 1 and voice 2 noteheads share the same x position
         when they're at the same beat, with opposing stems for distinction. -->
    <voice1XOffset>0</voice1XOffset>
    <voice2XOffset>0</voice2XOffset>
    <voice3XOffset>0</voice3XOffset>
    <voice4XOffset>0</voice4XOffset>

    <!-- Merge unison voices: when two voices have the same pitch at the same beat,
         draw a single shared notehead with stems going both directions. -->
    <mergeMatchingRests>true</mergeMatchingRests>

    <!-- Smaller minimum note distance so layout doesn't push notes apart unnecessarily -->
    <minNoteDistance>0.5</minNoteDistance>
  </Style>
</museScore>
