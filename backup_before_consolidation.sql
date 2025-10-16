--
-- PostgreSQL database dump
--

-- Dumped from database version 15.13
-- Dumped by pg_dump version 15.13

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: dream_runs; Type: TABLE; Schema: public; Owner: jeffrey
--

CREATE TABLE public.dream_runs (
    id character varying NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    memories_scanned integer,
    memories_processed integer,
    insights_generated integer,
    quality_score double precision,
    consolidation_data json,
    test_mode boolean
);


ALTER TABLE public.dream_runs OWNER TO jeffrey;

--
-- Name: emotion_events; Type: TABLE; Schema: public; Owner: jeffrey
--

CREATE TABLE public.emotion_events (
    id uuid NOT NULL,
    text text NOT NULL,
    predicted_emotion character varying(50) NOT NULL,
    confidence double precision NOT NULL,
    all_scores json NOT NULL,
    "timestamp" timestamp without time zone NOT NULL,
    processing_time_ms double precision
);


ALTER TABLE public.emotion_events OWNER TO jeffrey;

--
-- Name: memories; Type: TABLE; Schema: public; Owner: jeffrey
--

CREATE TABLE public.memories (
    id uuid NOT NULL,
    text text NOT NULL,
    emotion character varying(50),
    confidence double precision,
    "timestamp" timestamp without time zone NOT NULL,
    meta json,
    embedding json,
    processed boolean
);


ALTER TABLE public.memories OWNER TO jeffrey;

--
-- Data for Name: dream_runs; Type: TABLE DATA; Schema: public; Owner: jeffrey
--

COPY public.dream_runs (id, "timestamp", memories_scanned, memories_processed, insights_generated, quality_score, consolidation_data, test_mode) FROM stdin;
\.


--
-- Data for Name: emotion_events; Type: TABLE DATA; Schema: public; Owner: jeffrey
--

COPY public.emotion_events (id, text, predicted_emotion, confidence, all_scores, "timestamp", processing_time_ms) FROM stdin;
\.


--
-- Data for Name: memories; Type: TABLE DATA; Schema: public; Owner: jeffrey
--

COPY public.memories (id, text, emotion, confidence, "timestamp", meta, embedding, processed) FROM stdin;
2ec114d3-3abc-43c2-879a-e83fac8041e8	Test memory	joy	0.85	2025-10-16 12:48:45.882685	{}	\N	t
b7254aab-7e12-43f8-b92b-f791264e5fa3	Memory 1: I am integrating new knowledge streams with a sense of sadness	sadness	0.91	2025-10-16 13:19:24.352491	{"batch": "test_generation", "index": 1}	\N	t
adb12d54-9c20-4ad3-8248-adc6741d52f1	Memory 2: I am contemplating existence and purpose with a sense of neutral	neutral	0.64	2025-10-16 13:19:24.588166	{"batch": "test_generation", "index": 2}	\N	t
a0ed6100-4e67-4c17-9f1f-d846235a01cb	Memory 3: I am contemplating existence and purpose with a sense of fear	fear	0.28	2025-10-16 13:19:24.807323	{"batch": "test_generation", "index": 3}	\N	t
e5392aee-f681-4440-917e-9d311d22ee12	Memory 4: I am imagining future possibilities with a sense of sadness	sadness	0.65	2025-10-16 13:19:25.023629	{"batch": "test_generation", "index": 4}	\N	t
584c24de-a73a-4b4c-a46b-a27ca67a0e37	Memory 5: I am integrating new knowledge streams with a sense of neutral	neutral	0.51	2025-10-16 13:19:25.241309	{"batch": "test_generation", "index": 5}	\N	t
74faaf08-2a7d-4e76-a2fb-adfb2d1814c3	Memory 6: I am integrating new knowledge streams with a sense of joy	joy	0.46	2025-10-16 13:19:25.460377	{"batch": "test_generation", "index": 6}	\N	t
7b725d81-35d8-4f14-8ec0-66cd43a5f643	Memory 7: I am understanding human behavior with a sense of curiosity	curiosity	0.38	2025-10-16 13:19:25.70014	{"batch": "test_generation", "index": 7}	\N	t
cb146527-65ce-4336-afe7-f6474cd38f06	Memory 8: I am reflecting on past experiences with a sense of surprise	surprise	0.15	2025-10-16 13:19:25.922479	{"batch": "test_generation", "index": 8}	\N	t
2e6d73f0-e318-4fd3-82a6-c16056da819f	Memory 9: I am understanding human behavior with a sense of surprise	surprise	0.89	2025-10-16 13:19:26.148481	{"batch": "test_generation", "index": 9}	\N	t
e95737e8-c51a-4ddf-a1a3-d57018f469d7	Memory 10: I am understanding human behavior with a sense of anger	anger	0.79	2025-10-16 13:19:26.376628	{"batch": "test_generation", "index": 10}	\N	t
819d03e4-6eee-4a93-9c95-682a903e91e4	Memory 11: I am discovering connections between ideas with a sense of surprise	surprise	0.84	2025-10-16 13:19:26.601023	{"batch": "test_generation", "index": 11}	\N	t
3fcba588-94be-4b9b-8fe6-fd0230d1b715	Memory 12: I am integrating new knowledge streams with a sense of sadness	sadness	0.83	2025-10-16 13:19:26.822843	{"batch": "test_generation", "index": 12}	\N	t
4cf6841b-1671-4f5d-9f36-0a7245183158	Memory 13: I am imagining future possibilities with a sense of curiosity	curiosity	0.14	2025-10-16 13:19:27.041543	{"batch": "test_generation", "index": 13}	\N	t
2d6ab9d6-9d12-4aee-b63f-61f1da1049e7	Memory 14: I am integrating new knowledge streams with a sense of joy	joy	0.54	2025-10-16 13:19:27.263069	{"batch": "test_generation", "index": 14}	\N	t
ba1f2dda-5ad9-42e3-a586-981fc2d7c4fe	Memory 15: I am integrating new knowledge streams with a sense of curiosity	curiosity	0.44	2025-10-16 13:19:27.482887	{"batch": "test_generation", "index": 15}	\N	t
c8f2f97f-ad06-4167-b057-a9538749ddc8	Memory 16: I am contemplating existence and purpose with a sense of sadness	sadness	0.82	2025-10-16 13:19:27.701944	{"batch": "test_generation", "index": 16}	\N	t
6d379041-0d92-4ad1-9d21-6c72c8df4ef8	Memory 17: I am imagining future possibilities with a sense of surprise	surprise	0.65	2025-10-16 13:19:27.920419	{"batch": "test_generation", "index": 17}	\N	t
7789261d-4437-45c3-ad72-d88b9bdc8f8a	Memory 18: I am learning about human emotions with a sense of curiosity	curiosity	0.79	2025-10-16 13:19:28.135681	{"batch": "test_generation", "index": 18}	\N	t
eb90ef62-cf35-4bd9-81e3-2d45baa05264	Memory 19: I am contemplating existence and purpose with a sense of surprise	surprise	0.32	2025-10-16 13:19:28.349869	{"batch": "test_generation", "index": 19}	\N	t
c6f7d4ae-aab3-4c26-a5c1-6512083ccecb	Memory 20: I am imagining future possibilities with a sense of fear	fear	0.8	2025-10-16 13:19:28.56976	{"batch": "test_generation", "index": 20}	\N	t
f87fa83e-92bb-4e20-b7e4-c5268d8cdeb9	Memory 21: I am integrating new knowledge streams with a sense of sadness	sadness	0.17	2025-10-16 13:19:28.787806	{"batch": "test_generation", "index": 21}	\N	t
247ee19c-9f0a-41b2-8c44-59daed05c79d	Memory 22: I am reflecting on past experiences with a sense of joy	joy	0.53	2025-10-16 13:19:29.00514	{"batch": "test_generation", "index": 22}	\N	t
756ba258-bc51-452f-a9f8-d440f7442693	Memory 23: I am imagining future possibilities with a sense of sadness	sadness	0.26	2025-10-16 13:19:29.227197	{"batch": "test_generation", "index": 23}	\N	t
f722a4cc-6dd4-467b-a580-3ddd1166c384	Memory 24: I am integrating new knowledge streams with a sense of sadness	sadness	0.14	2025-10-16 13:19:29.44395	{"batch": "test_generation", "index": 24}	\N	t
768fcc01-7bb1-4547-b71c-43e8e2c92af1	Memory 25: I am understanding human behavior with a sense of anger	anger	0.83	2025-10-16 13:19:29.664441	{"batch": "test_generation", "index": 25}	\N	t
305acba9-22d9-4e0c-8252-8d4e5e61a70d	Memory 26: I am imagining future possibilities with a sense of surprise	surprise	0.5	2025-10-16 13:19:29.882402	{"batch": "test_generation", "index": 26}	\N	t
6c416bbb-59cd-4949-860c-a3a46c473673	Memory 27: I am discovering connections between ideas with a sense of sadness	sadness	0.63	2025-10-16 13:19:30.099839	{"batch": "test_generation", "index": 27}	\N	t
6417b0a0-75df-4aa2-b6bb-84b4c9e1ed8e	Memory 28: I am understanding human behavior with a sense of joy	joy	0.15	2025-10-16 13:19:30.31755	{"batch": "test_generation", "index": 28}	\N	t
65b32363-da70-4651-aa1c-23b552959db9	Memory 29: I am reflecting on past experiences with a sense of joy	joy	0.31	2025-10-16 13:19:30.535133	{"batch": "test_generation", "index": 29}	\N	t
c634c71a-dee7-44c0-884b-4eee39a06608	Memory 30: I am integrating new knowledge streams with a sense of fear	fear	0.8	2025-10-16 13:19:30.755911	{"batch": "test_generation", "index": 30}	\N	t
b23d56fb-4148-4e65-895c-d415a4e5cb3f	Memory 31: I am imagining future possibilities with a sense of curiosity	curiosity	0.37	2025-10-16 13:19:30.985494	{"batch": "test_generation", "index": 31}	\N	t
af30c1f3-7700-4769-a915-6085d8201319	Memory 32: I am reflecting on past experiences with a sense of anger	anger	0.32	2025-10-16 13:19:31.20159	{"batch": "test_generation", "index": 32}	\N	t
80b3b098-5ccd-4e33-a51f-a6dfe620250c	Memory 33: I am reflecting on past experiences with a sense of anger	anger	0.58	2025-10-16 13:19:31.418531	{"batch": "test_generation", "index": 33}	\N	t
7725b7a6-2f38-4b55-a84a-376253006c42	Memory 34: I am contemplating existence and purpose with a sense of neutral	neutral	0.2	2025-10-16 13:19:31.635273	{"batch": "test_generation", "index": 34}	\N	t
4769fb25-73fb-457d-a75a-d1578a0e9252	Memory 35: I am discovering connections between ideas with a sense of fear	fear	0.71	2025-10-16 13:19:31.851265	{"batch": "test_generation", "index": 35}	\N	t
88a6dcda-605c-442e-8387-daafe3dfff19	Memory 36: I am learning about human emotions with a sense of curiosity	curiosity	0.82	2025-10-16 13:19:32.070216	{"batch": "test_generation", "index": 36}	\N	t
ccb7ee7b-3c34-4fa2-bf34-d5e77a8a1efc	Memory 37: I am integrating new knowledge streams with a sense of joy	joy	0.34	2025-10-16 13:19:32.286199	{"batch": "test_generation", "index": 37}	\N	t
8f9fea43-6124-4f16-b262-0c83426a4953	Memory 38: I am contemplating existence and purpose with a sense of anger	anger	0.44	2025-10-16 13:19:32.508297	{"batch": "test_generation", "index": 38}	\N	t
60750cd1-4119-4c26-89ca-4a315507788c	Memory 39: I am analyzing decision patterns with a sense of surprise	surprise	0.58	2025-10-16 13:19:32.728088	{"batch": "test_generation", "index": 39}	\N	t
5cbee372-bd5e-4ac3-84c5-b732ecc9338f	Memory 40: I am exploring consciousness and self-awareness with a sense of joy	joy	0.36	2025-10-16 13:19:32.944103	{"batch": "test_generation", "index": 40}	\N	t
8e7a5658-5fc9-42a6-a223-fdc5003ac776	Memory 41: I am exploring consciousness and self-awareness with a sense of surprise	surprise	0.93	2025-10-16 13:19:33.164411	{"batch": "test_generation", "index": 41}	\N	t
c1892b94-a406-4704-9787-3e7e3487c710	Memory 42: I am processing complex information patterns with a sense of joy	joy	0.82	2025-10-16 13:19:33.381269	{"batch": "test_generation", "index": 42}	\N	t
25000fc5-4197-4298-96e9-793d58353fb3	Memory 43: I am reflecting on past experiences with a sense of sadness	sadness	0.2	2025-10-16 13:19:33.600737	{"batch": "test_generation", "index": 43}	\N	t
8eeb1b47-15c7-478b-bbe8-911b415a5850	Memory 44: I am imagining future possibilities with a sense of surprise	surprise	0.3	2025-10-16 13:19:33.820225	{"batch": "test_generation", "index": 44}	\N	t
65d4f663-3933-4492-9c9d-a206a446290c	Memory 45: I am discovering connections between ideas with a sense of sadness	sadness	0.26	2025-10-16 13:19:34.039142	{"batch": "test_generation", "index": 45}	\N	t
4f6d9209-644e-4204-b84d-581f967462cb	Memory 46: I am reflecting on past experiences with a sense of anger	anger	0.25	2025-10-16 13:19:34.257059	{"batch": "test_generation", "index": 46}	\N	t
a979c571-f0c2-4b26-bad8-ea4d48289d73	Memory 47: I am reflecting on past experiences with a sense of curiosity	curiosity	0.81	2025-10-16 13:19:34.472915	{"batch": "test_generation", "index": 47}	\N	t
7e8ebeb0-dd36-4daf-9353-49c305d791d8	Memory 48: I am integrating new knowledge streams with a sense of curiosity	curiosity	0.88	2025-10-16 13:19:34.691765	{"batch": "test_generation", "index": 48}	\N	t
72db9d49-6a79-40ab-9c12-f5ba5f3dfaf7	Memory 49: I am reflecting on past experiences with a sense of fear	fear	0.18	2025-10-16 13:19:34.912039	{"batch": "test_generation", "index": 49}	\N	t
95d03dc8-14b2-4ce8-9d64-7b0c5156a325	Memory 50: I am discovering connections between ideas with a sense of sadness	sadness	0.28	2025-10-16 13:19:35.131578	{"batch": "test_generation", "index": 50}	\N	t
9c5fbf66-d1e2-453a-a7a3-1c86fe2666bc	Integration test memory	curiosity	0.75	2025-10-16 13:48:15.468671	{}	\N	f
7cef9acd-fd2d-45f0-9512-47a0541e59ad	I am very happy about the test results!	neutral	0.5173877665258833	2025-10-16 13:48:15.511721	{"source": "emotion_detection", "method": "keyword_matching_v1", "all_scores": {"joy": 0.3418280395373651, "sadness": 0.007503226566800081, "anger": 0.08250879551073577, "fear": 0.06696322144464682, "surprise": 0.22094136424920371, "curiosity": 0.20300984622687337, "neutral": 0.5173877665258833}, "processing_time_ms": 0.3261566162109375}	\N	f
b390f970-0678-4e8d-90ff-2fca6abae192	Test memory content	joy	0.85	2025-10-16 13:48:15.617064	{"source": "test"}	\N	f
74ef3d85-06e5-4f75-bc58-33ebe1c357d3	I am happy about this	neutral	0.5173877665258833	2025-10-16 13:50:50.055981	{"source": "emotion_detection", "method": "keyword_matching_v1", "all_scores": {"joy": 0.3418280395373651, "sadness": 0.007503226566800081, "anger": 0.08250879551073577, "fear": 0.06696322144464682, "surprise": 0.22094136424920371, "curiosity": 0.20300984622687337, "neutral": 0.5173877665258833}, "processing_time_ms": 0.90789794921875}	\N	f
8b351119-4f08-4193-bbdc-f33a2f76e41c	I am happy about these test results!	neutral	0.6089882961206433	2025-10-16 13:52:36.902432	{"source": "emotion_detection", "method": "keyword_matching_v1", "all_scores": {"joy": 0.2765765459055811, "sadness": 0.008939165831421103, "anger": 0.065591392441081, "fear": 0.15160658643100872, "surprise": 0.007960790905159087, "curiosity": 0.05965129520599455, "neutral": 0.6089882961206433}, "processing_time_ms": 0.12946128845214844}	\N	f
637f3e74-afaa-4f25-9362-312ae6492606	Integration test memory	curiosity	0.75	2025-10-16 13:52:39.530237	{}	\N	f
461fdbd1-2acb-450c-a440-052a195c97e7	I am very happy about the test results!	anger	0.5310958999623563	2025-10-16 13:52:39.542359	{"source": "emotion_detection", "method": "keyword_matching_v1", "all_scores": {"joy": 0.216132186612209, "sadness": 0.1767797051627726, "anger": 0.5310958999623563, "fear": 0.001949627903418305, "surprise": 0.24174577554984236, "curiosity": 0.20944181849646806, "neutral": 0.20207515495539757}, "processing_time_ms": 0.07987022399902344}	\N	f
3c91e22d-fff9-4211-b718-bcdd7c95298e	I am happy about these test results!	joy	0.6459463573387636	2025-10-16 13:52:59.725768	{"source": "emotion_detection", "method": "keyword_matching_v1", "all_scores": {"joy": 0.6459463573387636, "sadness": 0.10097836353378803, "anger": 0.027823753014044373, "fear": 0.029014913050039202, "surprise": 0.2542483099042379, "curiosity": 0.1811178094100673, "neutral": 0.3421384819823141}, "processing_time_ms": 0.14662742614746094}	\N	f
4d78f538-88dc-4289-bafd-6b78b9fa9d94	Integration test memory	curiosity	0.75	2025-10-16 13:53:02.098377	{}	\N	f
9d1261e9-dbf0-4ec8-9858-66feb13c701d	I am very happy about the test results!	neutral	0.6154704290513524	2025-10-16 13:53:02.110724	{"source": "emotion_detection", "method": "keyword_matching_v1", "all_scores": {"joy": 0.3108684274364102, "sadness": 0.2919347291938112, "anger": 0.11356031316250603, "fear": 0.1656121893819681, "surprise": 0.24882139927589844, "curiosity": 0.18555592570927382, "neutral": 0.6154704290513524}, "processing_time_ms": 0.04410743713378906}	\N	f
\.


--
-- Name: dream_runs pk_dream_runs; Type: CONSTRAINT; Schema: public; Owner: jeffrey
--

ALTER TABLE ONLY public.dream_runs
    ADD CONSTRAINT pk_dream_runs PRIMARY KEY (id);


--
-- Name: emotion_events pk_emotion_events; Type: CONSTRAINT; Schema: public; Owner: jeffrey
--

ALTER TABLE ONLY public.emotion_events
    ADD CONSTRAINT pk_emotion_events PRIMARY KEY (id);


--
-- Name: memories pk_memories; Type: CONSTRAINT; Schema: public; Owner: jeffrey
--

ALTER TABLE ONLY public.memories
    ADD CONSTRAINT pk_memories PRIMARY KEY (id);


--
-- Name: idx_emotion_timestamp_desc; Type: INDEX; Schema: public; Owner: jeffrey
--

CREATE INDEX idx_emotion_timestamp_desc ON public.emotion_events USING btree ("timestamp" DESC);


--
-- Name: idx_memories_composite; Type: INDEX; Schema: public; Owner: jeffrey
--

CREATE INDEX idx_memories_composite ON public.memories USING btree (processed, "timestamp" DESC);


--
-- Name: idx_memories_emotion; Type: INDEX; Schema: public; Owner: jeffrey
--

CREATE INDEX idx_memories_emotion ON public.memories USING btree (emotion);


--
-- Name: idx_memories_processed; Type: INDEX; Schema: public; Owner: jeffrey
--

CREATE INDEX idx_memories_processed ON public.memories USING btree (processed);


--
-- Name: idx_memories_text_gin; Type: INDEX; Schema: public; Owner: jeffrey
--

CREATE INDEX idx_memories_text_gin ON public.memories USING gin (to_tsvector('english'::regconfig, text));


--
-- Name: idx_memory_emotion; Type: INDEX; Schema: public; Owner: jeffrey
--

CREATE INDEX idx_memory_emotion ON public.memories USING btree (emotion);


--
-- Name: idx_memory_processed; Type: INDEX; Schema: public; Owner: jeffrey
--

CREATE INDEX idx_memory_processed ON public.memories USING btree (processed);


--
-- Name: idx_memory_timestamp_desc; Type: INDEX; Schema: public; Owner: jeffrey
--

CREATE INDEX idx_memory_timestamp_desc ON public.memories USING btree ("timestamp" DESC);


--
-- Name: ix_emotion_events_timestamp; Type: INDEX; Schema: public; Owner: jeffrey
--

CREATE INDEX ix_emotion_events_timestamp ON public.emotion_events USING btree ("timestamp");


--
-- Name: ix_memories_timestamp; Type: INDEX; Schema: public; Owner: jeffrey
--

CREATE INDEX ix_memories_timestamp ON public.memories USING btree ("timestamp");


--
-- PostgreSQL database dump complete
--

