// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/openai/openai-go/option"
)

func TestAudioModelsRegistered(t *testing.T) {
	plugin := &OpenAI{APIKey: "test-key"}
	actions := plugin.Init(context.Background())
	byName := make(map[string]api.Action, len(actions))
	for _, action := range actions {
		byName[action.Name()] = action
	}

	for _, name := range []string{
		"openai/tts-1",
		"openai/tts-1-hd",
		"openai/gpt-4o-mini-tts",
		"openai/whisper-1",
		"openai/gpt-4o-transcribe",
		"openai/gpt-4o-mini-transcribe",
	} {
		if byName[name] == nil {
			t.Errorf("model %q was not registered", name)
		}
	}

	miniTTS := byName["openai/gpt-4o-mini-tts"].Desc()
	modelMetadata, ok := miniTTS.Metadata["model"].(map[string]any)
	if !ok {
		t.Fatalf("model metadata = %#v, want map", miniTTS.Metadata["model"])
	}
	supports, ok := modelMetadata["supports"].(map[string]any)
	if !ok {
		t.Fatalf("supports metadata = %#v, want map", modelMetadata["supports"])
	}
	output, ok := supports["output"].([]string)
	if !ok || len(output) != 1 || output[0] != "media" {
		t.Errorf("TTS output support = %#v, want [media]", supports["output"])
	}
}

func TestGPT4oMiniTTSSchemaOmitsSpeed(t *testing.T) {
	schema := supportedSpeechModels["gpt-4o-mini-tts"].ConfigSchema
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema properties = %#v, want map", schema["properties"])
	}
	if _, ok := properties["speed"]; ok {
		t.Error("gpt-4o-mini-tts schema unexpectedly includes speed")
	}
}

func TestWhisperSchemaIncludesTranslate(t *testing.T) {
	schema := supportedWhisperModels["whisper-1"].ConfigSchema
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		t.Fatalf("schema properties = %#v, want map", schema["properties"])
	}
	translate, ok := properties["translate"].(map[string]any)
	if !ok {
		t.Fatalf("translate schema = %#v, want map", properties["translate"])
	}
	if got := translate["type"]; got != "boolean" {
		t.Errorf("translate type = %v, want boolean", got)
	}
	if got := translate["default"]; got != false {
		t.Errorf("translate default = %v, want false", got)
	}
}

func TestWhisperModelTranslation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/audio/translations" {
			t.Errorf("path = %q, want /audio/translations", r.URL.Path)
		}
		reader, err := r.MultipartReader()
		if err != nil {
			t.Fatalf("MultipartReader: %v", err)
		}
		fields := map[string]string{}
		for {
			part, err := reader.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Fatalf("NextPart: %v", err)
			}
			data, err := io.ReadAll(part)
			if err != nil {
				t.Fatalf("read part: %v", err)
			}
			fields[part.FormName()] = string(data)
		}
		for key, want := range map[string]string{
			"model":           "whisper-1",
			"prompt":          "Names: Genkit",
			"response_format": "text",
			"file":            "audio",
		} {
			if got := fields[key]; got != want {
				t.Errorf("request[%q] = %q, want %q", key, got, want)
			}
		}
		if _, ok := fields["translate"]; ok {
			t.Error("translate control field was sent to OpenAI")
		}
		w.Header().Set("Content-Type", "text/plain")
		_, _ = io.WriteString(w, "Hello in English")
	}))
	t.Cleanup(server.Close)

	plugin := &OpenAI{
		APIKey: "test-key",
		Opts:   []option.RequestOption{option.WithBaseURL(server.URL)},
	}
	actions := plugin.Init(context.Background())
	var whisper ai.Model
	for _, action := range actions {
		if action.Name() == "openai/whisper-1" {
			whisper = action.(ai.Model)
			break
		}
	}
	if whisper == nil {
		t.Fatal("openai/whisper-1 was not registered")
	}
	resp, err := whisper.Generate(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserMessage(
			ai.NewTextPart("Names: Genkit"),
			ai.NewMediaPart("audio/wav", "data:audio/wav;base64,YXVkaW8="),
		)},
		Config: map[string]any{
			"response_format": "text",
			"translate":       true,
		},
	}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := resp.Text(); got != "Hello in English" {
		t.Errorf("response text = %q, want Hello in English", got)
	}
}

func TestListActionsClassifiesAudioModels(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"object":"list","data":[`+
			`{"id":"future-tts","object":"model","created":1,"owned_by":"openai"},`+
			`{"id":"future-transcribe","object":"model","created":1,"owned_by":"openai"}]}`)
	}))
	t.Cleanup(server.Close)
	plugin := &OpenAI{
		APIKey: "test-key",
		Opts:   []option.RequestOption{option.WithBaseURL(server.URL)},
	}
	plugin.Init(context.Background())
	actions := plugin.ListActions(context.Background())
	if len(actions) != 2 {
		t.Fatalf("ListActions() returned %d actions, want 2", len(actions))
	}
	for _, action := range actions {
		modelMetadata := action.Metadata["model"].(map[string]any)
		supports := modelMetadata["supports"].(map[string]any)
		output := supports["output"].([]string)
		if action.Name == "openai/future-tts" && output[0] != "media" {
			t.Errorf("TTS output = %#v, want media", output)
		}
		if action.Name == "openai/future-transcribe" && (output[0] != "text" || supports["media"] != true) {
			t.Errorf("transcription supports = %#v, want media input and text output", supports)
		}
	}
}

func TestResolveAudioModels(t *testing.T) {
	plugin := &OpenAI{APIKey: "test-key"}
	plugin.Init(context.Background())
	for _, tc := range []struct {
		name       string
		wantOutput string
		wantMedia  bool
	}{
		{name: "custom-tts", wantOutput: "media"},
		{name: "custom-transcribe", wantOutput: "text", wantMedia: true},
		{name: "whisper-future", wantOutput: "text", wantMedia: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			action := plugin.ResolveAction(api.ActionTypeModel, tc.name)
			if action == nil {
				t.Fatal("ResolveAction() = nil")
			}
			modelMetadata := action.Desc().Metadata["model"].(map[string]any)
			supports := modelMetadata["supports"].(map[string]any)
			if got := supports["media"]; got != tc.wantMedia {
				t.Errorf("media support = %v, want %v", got, tc.wantMedia)
			}
			output := supports["output"].([]string)
			if len(output) == 0 || output[0] != tc.wantOutput {
				t.Errorf("output support = %#v, want first value %q", output, tc.wantOutput)
			}
		})
	}
}
