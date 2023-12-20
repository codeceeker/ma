package com.example.fragapp;

import android.os.Bundle;

import androidx.fragment.app.Fragment;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;


public class FragementOne extends Fragment {




    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState)
    {
        // Inflate the layout for this fragment
        Toast.makeText(getActivity(),"Hello I am Fragment!",Toast.LENGTH_SHORT).show();
        return inflater.inflate(R.layout.fragment_fragement_one, container, false);
    }
}