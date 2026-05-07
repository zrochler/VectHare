/**
 * ============================================================================
 * VECTHARE SHARED VECTORIZATION UTILITIES
 * ============================================================================
 * Unified enrichment pipeline for both chat and content vectorization.
 * Single source of truth for hashing, keyword extraction, and chat utilities.
 *
 * @author Coneja Chibi
 * @version 1.0.0
 * ============================================================================
 */

import { getStringHash as calculateHash } from '../../../../utils.js';
import {  substituteParams } from '../../../../../script.js';

import { extractLorebookKeywords, extractTextKeywords, extractBM25Keywords } from './keyword-boost.js';
import { cleanText } from './text-cleaning.js';
import { LRUCache } from '../utils/data-structures.js';

const HASH_CACHE_SIZE = 10000;
const hashCache = new LRUCache(HASH_CACHE_SIZE);

/**
 * Gets the hash value for a string (with LRU caching)
 * @param {string} str Input string
 * @returns {number} Hash value
 */
export function getStringHash(str) {
    const cached = hashCache.get(str);
    if (cached !== undefined) {
        return cached;
    }
    const hash = calculateHash(str);
    hashCache.set(str, hash);
    return hash;
}

/**
 * Gets message text without file attachments
 * Matches behavior of ST vectors extension for hash compatibility
 * @param {object} message Chat message object
 * @returns {string} Message text without attachment prefix
 */
export function getTextWithoutAttachments(message) {
    const fileLength = message?.extra?.fileLength || 0;
    return String(message?.mes || '').substring(fileLength).trim();
}

/**
 * Enriches vector items with hashes and keywords
 * Unified enrichment for both chat and content vectorization pipelines
 *
 * @param {object[]} items - Array of items to enrich, each with {text, index, metadata?, ...}
 * @param {object} settings - VectHare settings
 *   - keywordLevel: 'off', 'minimal', 'balanced', 'aggressive'
 *   - keywordBaseWeight: base weight for keywords
 * @param {object} context - Context-specific enrichment options
 *   - contentType: 'chat', 'lorebook', 'character', 'document', 'url', etc.
 *   - preparedContent: prepared content data (entries, character, etc.)
 *   - vecthareSettings: full extension settings (includes custom_stopwords)
 * @returns {object[]} Enriched items with hash and keywords added
 */
export function enrichVectorItems(items, settings, context = {}) {
    const {
        keywordLevel = 'balanced',
        keywordBaseWeight = 1.5,
    } = settings;

    const {
        contentType = 'generic',
        preparedContent = {},
        vecthareSettings = {},
    } = context;

    return items.map((item, index) => {
        const itemText = typeof item === 'string' ? item : (item.text || '');
        let keywords = [];
        let entryName = null;
        let entryUid = null;

        // Extract keywords based on content type
        if (contentType === 'lorebook' && preparedContent.entries?.[index]) {
            // Lorebook: use trigger keys + auto-extracted keywords
            const entry = preparedContent.entries[index];
            entryName = entry.comment || entry.name || entry.key?.[0] || 'Entry';
            entryUid = entry.uid;

            // Get explicit trigger keys (manually set, so use base weight)
            const triggerKeys = extractLorebookKeywords(entry, vecthareSettings);
            keywords = triggerKeys.map(k => ({ text: k, weight: keywordBaseWeight }));

            // Also get auto-extracted keywords with frequency-based weights
            if (keywordLevel !== 'off') {
                const autoKeywords = extractTextKeywords(entry.content || itemText, {
                    level: keywordLevel,
                    baseWeight: keywordBaseWeight,
                    settings: vecthareSettings,
                });
                keywords = keywords.concat(autoKeywords);
            }
        } else if (contentType === 'chat') {
            // Chat: use BM25/TF-IDF for distinctive words
            if (keywordLevel !== 'off') {
                keywords = extractBM25Keywords(itemText, {
                    level: keywordLevel,
                    baseWeight: keywordBaseWeight,
                    settings: vecthareSettings,
                });
            }
        } else if (contentType === 'character' && preparedContent.character?.name) {
            // Character: add character name as high-weight keyword
            if (keywordLevel !== 'off') {
                keywords = extractTextKeywords(itemText, {
                    level: keywordLevel,
                    baseWeight: keywordBaseWeight,
                    settings: vecthareSettings,
                });
            }
            // Add character name with bonus weight
            keywords.push({
                text: preparedContent.character.name.toLowerCase(),
                weight: keywordBaseWeight + 0.5,
            });
        } else {
            // Default: URL, wiki, document, YouTube, etc. - use frequency-based extraction
            if (keywordLevel !== 'off') {
                keywords = extractTextKeywords(itemText, {
                    level: keywordLevel,
                    baseWeight: keywordBaseWeight,
                    settings: vecthareSettings,
                });
            }
        }

        // Add speaker name as keyword for chat items
        if (contentType === 'chat' && item.metadata?.speaker) {
            keywords.push({
                text: item.metadata.speaker.toLowerCase(),
                weight: keywordBaseWeight,
            });
        }

        // Deduplicate keywords (keep highest weight for duplicates)
        const keywordMap = new Map();
        for (const kw of keywords) {
            const existing = keywordMap.get(kw.text);
            if (!existing || kw.weight > existing.weight) {
                keywordMap.set(kw.text, kw);
            }
        }
        const dedupedKeywords = Array.from(keywordMap.values());

        return {
            text: itemText,
            hash: getStringHash(itemText),
            index: item.index ?? index,
            keywords: dedupedKeywords,
            metadata: {
                contentType,
                entryName,
                entryUid,
                keywordLevel,
                keywordBaseWeight,
                ...(item.metadata || {}),
            },
        };
    });
}



/**
 * Expands an ILS Summary message into its constituent OriginalMessages plus
 * the summary itself, with virtual indices accounting for expansion.
 *
 * ILS Summaries are inline summary messages (name === 'Summary') that store
 * their original messages in extra.ILS_Data.OriginalMessages. When vectorizing,
 * we want both the originals AND the summary, with correct sequential indices.
 *
 * Example: ILS Summary at virtualIndex 0 covering 3 OriginalMessages:
 *   → OriginalMessage[0] gets virtualIndex 0
 *   → OriginalMessage[1] gets virtualIndex 1
 *   → OriginalMessage[2] gets virtualIndex 2
 *   → Summary gets virtualIndex 3
 *   → nextVirtualIndex returned = 4
 *
 * @param {object} msg - Raw chat message from context.chat
 * @param {number} virtualIndex - The starting virtual index for this message
 * @returns {{ expandedMessages: object[], nextVirtualIndex: number }}
 *   expandedMessages: array of normalized message objects (originals first, summary last)
 *   nextVirtualIndex: the next available virtual index after expansion
 */
export function expandILSMessage(msg, virtualIndex) {
    const ilsData = msg?.extra?.ILS_Data;
    const originalMessages = ilsData?.OriginalMessages;

    // Not an ILS Summary — return as-is with a single virtual index
    if (!originalMessages || !Array.isArray(originalMessages) || originalMessages.length === 0) {
        return {
            expandedMessages: [msg],
            nextVirtualIndex: virtualIndex + 1,
        };
    }

    console.log(`[VectHare ILS] Expanding ILS Summary at virtualIndex ${virtualIndex} with ${originalMessages.length} OriginalMessages`);

    const expanded = [];
    let idx = virtualIndex;

    // 1. Expand OriginalMessages in order
    for (const orig of originalMessages) {
        const rawText = String(substituteParams(orig.mes || ''));
        const text = cleanText(rawText);

        if (orig.extra?.ILS_Data?.OriginalMessages) {
                // ILS Summaries can be nested, so recursively expand if we encounter another summary in the originals
                const { expandedMessages, nextVirtualIndex } = expandILSMessage(orig, virtualIndex);
                expanded.push(...expandedMessages);
                idx = nextVirtualIndex;
            } else {
                expanded.push({
                    text,
                    hash: getStringHash(substituteParams(getTextWithoutAttachments(orig))),
                    index: idx,
                    is_user: orig.is_user ?? false,
                    name: orig.name,
                    // Mark that this came from ILS expansion for traceability
                    isILSOriginal: true,
                });
                idx++;
            }
    }

    // 2. Append the summary itself AFTER its originals
    const summaryText = String(substituteParams(msg.mes || ''));
    const cleanedSummaryText = cleanText(summaryText);
    expanded.push({
        text: cleanedSummaryText,
        hash: getStringHash(substituteParams(getTextWithoutAttachments(msg))),
        index: idx,
        is_user: false,
        name: 'Summary',
        // Mark as ILS Summary so chunking strategies can keep it standalone
        isILSSummary: true,
    });
    idx++;

    return {
        expandedMessages: expanded,
        nextVirtualIndex: idx,
    };
}

/**
 * Clears the hash cache (useful for testing)
 */
export function clearHashCache() {
    hashCache.clear();
}
